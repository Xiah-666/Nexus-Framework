# nemesis_web_api.py - FastAPI Backend routing all UI operations via orchestrator/agents/plugins
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from pray import NemesisConfig, NemesisOrchestrator, PluginRegistry, execute_plugin
from alias_manager import AliasManager
from llm_agent_interface import LLMAgentInterface
import logging
from typing import List, Optional
from jose import JWTError, jwt
from datetime import datetime, timedelta
import asyncio
import requests

app = FastAPI(title="Nemesis-Nexus API", description="Unified API for CyberAgent LLM & Plugins (Red Team Edition)")

nemesis_config = NemesisConfig()
orchestrator = NemesisOrchestrator(nemesis_config)
llm_agent = LLMAgentInterface(nemesis_config, None)  # supply valid model_manager if needed

USERS_DB = {
    "admin": {"username": "admin", "hashed_password": "admin", "roles": ["admin", "operator", "viewer"]},
    "demo": {"username": "demo", "hashed_password": "demo", "roles": ["operator", "viewer"]},
}
SECRET_KEY = "CHANGE_ME"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 240

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def verify_password(plain: str, hashed: str) -> bool:
    return plain == hashed

def get_user(username: str):
    user = USERS_DB.get(username)
    return user

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

def require_roles(roles: List[str]):
    def role_checker(user = Depends(get_current_user)):
        user_roles = user["roles"]
        if not any(role in user_roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges ({roles} required)"
            )
        return user
    return role_checker

audit_logger = logging.getLogger('audit')

@app.on_event("startup")
async def startup_event():
    await orchestrator.initialize()

class PluginExecRequest(BaseModel):
    name: str
    options: dict = {}

class AgentTaskRequest(BaseModel):
    agent: str
    task: str
    context: dict = {}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        audit_logger.warning(f"Failed login for {form_data.username}")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user["username"]})
    audit_logger.info(f"Login success for {user['username']}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/whoami")
async def get_whoami(user=Depends(get_current_user)):
    return {"user": user["username"], "roles": user["roles"]}

# --- Main entrypoints ---
from pray import NemesisOrchestrator
import uuid
from collections import deque

# --- Operation Queue/Management (in-memory; you may want persistent for prod) ---
operation_queue = deque()
active_ops = {}
completed_ops = {}
operation_logs = {}
queue_lock = asyncio.Lock()  # Use async lock for FastAPI

class OperationStatus(BaseModel):
    id: str
    agent: str = None
    plugin: str = None
    task: str = None
    options: dict = {}
    context: dict = {}
    status: str = "queued"  # queued, running, completed, error, killed
    progress: float = 0.0
    result: Optional[dict|str] = None
    error: Optional[str] = None
    retask_of: Optional[str] = None
    logs: list = []
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

async def append_op_log(op_id, msg):
    async with queue_lock:
        log = operation_logs.setdefault(op_id, deque(maxlen=1000))
        rec = {"timestamp": datetime.utcnow().isoformat(), "message": msg}
        log.append(rec)
        if op_id in active_ops:
            active_ops[op_id]["logs"].append(rec)

# New task queue endpoint for plugin
@app.post("/ops/queue/plugin")
async def queue_plugin_op(req: PluginExecRequest, user=Depends(require_roles(['operator', 'admin']))):
    op_id = str(uuid.uuid4())
    op = OperationStatus(id=op_id, plugin=req.name, options=req.options, status="queued", logs=[], started_at=datetime.utcnow())
    # Queue op
    async with queue_lock:
        operation_queue.append(op.dict())
        active_ops[op_id] = op.dict()
    await append_op_log(op_id, f"Queued plugin op: {req.name}")
    return {"op_id": op_id, "status": "queued"}

@app.post("/ops/queue/agent")
async def queue_agent_op(req: AgentTaskRequest, user=Depends(require_roles(['operator', 'admin']))):
    op_id = str(uuid.uuid4())
    op = OperationStatus(id=op_id, agent=req.agent, task=req.task, context=req.context, status="queued", logs=[], started_at=datetime.utcnow())
    async with queue_lock:
        operation_queue.append(op.dict())
        active_ops[op_id] = op.dict()
    await append_op_log(op_id, f"Queued agent op: {req.agent}::{req.task}")
    return {"op_id": op_id, "status": "queued"}

@app.get("/ops/{op_id}/status")
async def get_op_status(op_id: str, user=Depends(require_roles(["viewer", "operator", "admin"]))):
    async with queue_lock:
        op = active_ops.get(op_id) or completed_ops.get(op_id)
        if not op:
            raise HTTPException(status_code=404, detail="Operation not found")
        return op

@app.get("/ops/{op_id}/logs")
async def get_op_logs(op_id: str, user=Depends(require_roles(["viewer", "operator", "admin"]))):
    async with queue_lock:
        logs = list(operation_logs.get(op_id, []))
        return {"op_id": op_id, "logs": logs}

@app.post("/ops/{op_id}/kill")
async def kill_op(op_id: str, user=Depends(require_roles(["admin"]))) -> dict:
    async with queue_lock:
        op = active_ops.get(op_id)
        if not op or op["status"] not in ("queued", "running"):
            raise HTTPException(status_code=400, detail="Cannot kill operation (not found or already completed)")
        op["status"] = "killed"
        await append_op_log(op_id, f"Operation killed by {user['username']}")
        return {"status": "killed", "op_id": op_id}

@app.post("/ops/{op_id}/retask")
async def retask_op(op_id: str, req: dict, user=Depends(require_roles(["operator", "admin"]))):
    async with queue_lock:
        old_op = active_ops.get(op_id) or completed_ops.get(op_id)
        if not old_op:
            raise HTTPException(status_code=404, detail="Original operation not found")
        new_op_id = str(uuid.uuid4())
        # For plugin
        if old_op.get("plugin"):
            new_op = OperationStatus(id=new_op_id, plugin=old_op["plugin"], options=req.get("options", old_op["options"]), task=old_op.get("task"), status="queued", logs=[], started_at=datetime.utcnow(), retask_of=op_id)
        elif old_op.get("agent"):
            new_op = OperationStatus(id=new_op_id, agent=old_op["agent"], task=req.get("task", old_op["task"]), context=req.get("context", old_op.get("context", {})), status="queued", logs=[], started_at=datetime.utcnow(), retask_of=op_id)
        else:
            raise HTTPException(status_code=400, detail="Unsupported operation type")
        operation_queue.append(new_op.dict())
        active_ops[new_op_id] = new_op.dict()
        await append_op_log(new_op_id, f"Retasked from op {op_id} by {user['username']}")
        return {"op_id": new_op_id, "status": "queued", "retask_of": op_id}

# In the background, a worker picks up from operation_queue and executes synchronously (demo only; production: use thread/asyncio/etc)
async def background_op_worker():
    while True:
        await asyncio.sleep(0.5)
        if not operation_queue:
            continue
        async with queue_lock:
            op = operation_queue.popleft()
            op_id = op["id"]
            if op["status"] != "queued":
                continue
            op["status"] = "running"
            active_ops[op_id] = op
        await append_op_log(op_id, "Started operation")
        try:
            if op.get("plugin"):
                result = execute_plugin(op["plugin"], **op.get("options", {}))
            elif op.get("agent"):
                agent = orchestrator.agents.get(op["agent"])
                if agent:
                    result = await agent.execute_task(op["task"], context=op.get("context", {}))
                else:
                    raise Exception(f"Agent '{op['agent']}' not found")
            else:
                raise Exception("Invalid operation type")
            async with queue_lock:
                op["result"] = result
                op["status"] = "completed"
                op["finished_at"] = datetime.utcnow()
                await append_op_log(op_id, "Operation completed")
                completed_ops[op_id] = op
                if op_id in active_ops:
                    del active_ops[op_id]
        except Exception as e:
            async with queue_lock:
                op["error"] = str(e)
                op["status"] = "error"
                op["finished_at"] = datetime.utcnow()
                completed_ops[op_id] = op
                await append_op_log(op_id, f"Operation error: {e}")
                if op_id in active_ops:
                    del active_ops[op_id]

@app.on_event("startup")
async def start_worker_event():
    asyncio.create_task(background_op_worker())

# Backward compatibility: keep original sync plugin/agent endpoints
@app.post("/plugin/exec")
async def plugin_exec(req: PluginExecRequest, user=Depends(require_roles(['operator', 'admin']))):
    try:
        result = execute_plugin(req.name, **req.options)
        audit_logger.info(f"User {user['username']} executed plugin {req.name} options={req.options}")
        return {"result": result}
    except Exception as e:
        audit_logger.error(f"User {user['username']} failed to execute plugin {req.name}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Agent execution stays as-is in agent_exec endpoint below.

# ================= Credential Vault API =================
from typing import Dict
from threading import Lock
from cryptography.fernet import Fernet

# In-memory encrypted credential storage (for demo, replace with database in production)
_cred_vault: Dict[str, Dict] = {}
_vault_lock = Lock()
_vault_key = Fernet.generate_key()
_vault_crypto = Fernet(_vault_key)

class VaultCredential(BaseModel):
    domain: str
    username: str
    secret: str

@app.get("/vault/credentials")
async def list_credentials(user=Depends(require_roles(["admin", "operator"]))) -> dict:
    # Return decrypted credentials for authorized users
    with _vault_lock:
        output = []
        for cid, data in _cred_vault.items():
            cred = dict(id=cid, domain=data['domain'], username=data['username'])
            # Only admin can see secret
            if 'admin' in user["roles"]:
                cred["secret"] = _vault_crypto.decrypt(data['secret']).decode()
            output.append(cred)
        return {"credentials": output}

@app.post("/vault/credentials")
async def store_credential(cred: VaultCredential, user=Depends(require_roles(["admin", "operator"]))) -> dict:
    import uuid
    cid = str(uuid.uuid4())
    with _vault_lock:
        _cred_vault[cid] = {
            'domain': cred.domain,
            'username': cred.username,
            'secret': _vault_crypto.encrypt(cred.secret.encode()),
        }
    audit_logger.info(f"User {user['username']} stored credential for {cred.domain}")
    return {"status": "stored", "id": cid}

@app.put("/vault/credentials/{cred_id}")
async def edit_credential(cred_id: str, payload: dict, user=Depends(require_roles(["admin", "operator"]))) -> dict:
    field = payload.get('field')
    value = payload.get('value')
    with _vault_lock:
        cred = _cred_vault.get(cred_id)
        if not cred:
            raise HTTPException(status_code=404, detail="Credential not found")
        if field == 'secret':
            cred['secret'] = _vault_crypto.encrypt(value.encode())
        else:
            cred[field] = value
    audit_logger.info(f"User {user['username']} edited credential {cred_id}")
    return {"status": "edited", "id": cred_id}

@app.delete("/vault/credentials/{cred_id}")
async def delete_credential(cred_id: str, user=Depends(require_roles(["admin"]))) -> dict:
    with _vault_lock:
        removed = _cred_vault.pop(cred_id, None)
    audit_logger.info(f"User {user['username']} deleted credential {cred_id}")
    return {"status": "deleted" if removed else "not found"}

# ================= Real-Time Visualization endpoint (live system status) =================

@app.get("/status/live")
async def get_live_status(user=Depends(require_roles(["viewer", "operator", "admin"]))):
    agents = list(orchestrator.agents.keys())
    plugins = list(PluginRegistry._plugins.keys())
    # Return live queued/running ops (from the async task system)
    async with queue_lock:
        currently_running = [dict(v) for v in active_ops.values() if v["status"] in ("queued", "running")]
    return {
        "agents": agents,
        "plugins": plugins,
        "active_ops": currently_running,
    }

@app.post("/agent/exec")
async def agent_exec(req: AgentTaskRequest, user=Depends(require_roles(['operator', 'admin']))):
    agent = orchestrator.agents.get(req.agent)
    if not agent:
        audit_logger.warning(f"User {user['username']} failed to exec agent '{req.agent}' (missing agent)")
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent}' not found.")
    result = await agent.execute_task(req.task, context=req.context)
    audit_logger.info(f"User {user['username']} executed agent {req.agent} task={req.task}")
    return {"result": result}

@app.get("/plugins")
def get_plugins(user=Depends(require_roles(["viewer", "operator", "admin"]))):
    audit_logger.info(f"User {user['username']} listed plugins")
    return {"plugins": list(PluginRegistry._plugins.keys())}

@app.get("/agents")
def get_agents(user=Depends(require_roles(["viewer", "operator", "admin"]))):
    audit_logger.info(f"User {user['username']} listed agents")
    return {"agents": list(orchestrator.agents.keys())}

@app.get("/aliases")
def get_aliases(user=Depends(require_roles(["viewer", "operator", "admin"]))):
    alias_mgr = AliasManager("config.yaml")
    audit_logger.info(f"User {user['username']} retrieved aliases")
    return {"aliases": alias_mgr.view_aliases()}

@app.post("/alias/create")
def api_create_alias(kind: str, name: str, value: str, user=Depends(require_roles(['operator', 'admin']))):
    alias_mgr = AliasManager("config.yaml")
    alias_mgr.create_alias(kind, name, value)
    audit_logger.info(f"User {user['username']} created alias {name}:{value}")
    return {"status": "created", "alias": {"type": kind, "name": name, "value": value}}

@app.post("/alias/delete")
def api_delete_alias(kind: str, name: str, user=Depends(require_roles(['operator', 'admin']))):
    alias_mgr = AliasManager("config.yaml")
    ok = alias_mgr.delete_alias(kind, name)
    audit_logger.info(f"User {user['username']} deleted alias {name}")
    return {"status": "deleted" if ok else "not found", "alias": {"type": kind, "name": name}}

class LLMChatRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None
    backend: Optional[str] = None
    model: Optional[str] = None

@app.post("/llm/chat")
async def llm_chat(req: LLMChatRequest, user=Depends(require_roles(['operator', 'admin']))):
    response = await llm_agent.generate(req.prompt, context=req.context, backend=req.backend, model=req.model)
    audit_logger.info(f"User {user['username']} initiated LLM chat (model={req.model})")
    return {"response": response}

# ================= Model Management Endpoints =================

@app.get("/models")
async def get_available_models(user=Depends(require_roles(["viewer", "operator", "admin"]))):
    """Get list of available models from Ollama"""
    try:
        # Get models from Ollama
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            ollama_models = response.json().get('models', [])
            
            # Format model list for UI
            available_models = []
            for model in ollama_models:
                model_name = model.get('name', '')
                if model_name:
                    available_models.append({
                        'name': model_name,
                        'size': model.get('size', 0),
                        'status': 'available',
                        'modified_at': model.get('modified_at', ''),
                        'digest': model.get('digest', '')
                    })
            
            audit_logger.info(f"User {user['username']} retrieved model list")
            return {"models": available_models}
        else:
            raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except Exception as e:
        audit_logger.error(f"Failed to get models for user {user['username']}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")

@app.get("/models/current")
async def get_current_model(user=Depends(require_roles(["viewer", "operator", "admin"]))):
    """Get currently selected model"""
    current_model = nemesis_config.default_model
    if hasattr(llm_agent, 'model_manager') and llm_agent.model_manager:
        # Try to get from model manager if available
        try:
            current_model = llm_agent.model_manager.config.default_model
        except:
            pass
    
    return {"current_model": current_model}

class ModelSelectionRequest(BaseModel):
    model_name: str

@app.post("/models/select")
async def select_model(req: ModelSelectionRequest, user=Depends(require_roles(["operator", "admin"]))):
    """Select active model for session"""
    try:
        # Validate that model exists
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            ollama_models = response.json().get('models', [])
            available_model_names = [model.get('name', '') for model in ollama_models]
            
            if req.model_name not in available_model_names:
                raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found")
            
            # Update default model in config
            nemesis_config.default_model = req.model_name
            
            # Update model manager if available
            if hasattr(llm_agent, 'model_manager') and llm_agent.model_manager:
                llm_agent.model_manager.config.default_model = req.model_name
            
            audit_logger.info(f"User {user['username']} selected model: {req.model_name}")
            return {"status": "success", "selected_model": req.model_name}
        else:
            raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except HTTPException:
        raise
    except Exception as e:
        audit_logger.error(f"Failed to select model for user {user['username']}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to select model: {str(e)}")

class ModelDownloadRequest(BaseModel):
    model_name: str

@app.post("/models/download")
async def download_model(req: ModelDownloadRequest, user=Depends(require_roles(["admin"]))):
    """Download a model from Ollama registry"""
    try:
        # Start download process
        import subprocess
        import asyncio
        
        audit_logger.info(f"User {user['username']} initiated download of model: {req.model_name}")
        
        # Start download in background
        process = await asyncio.create_subprocess_exec(
            'ollama', 'pull', req.model_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Return immediately while download continues in background
        return {"status": "download_started", "model_name": req.model_name, "message": "Download started in background"}
        
    except Exception as e:
        audit_logger.error(f"Failed to download model for user {user['username']}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model download: {str(e)}")
