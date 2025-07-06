# NEMESIS-NEXUS Auto UI Fallback System

This document explains the automatic UI fallback system implemented in NEMESIS-NEXUS, which automatically detects when X11/desktop GUI is not available and falls back to the Streamlit web interface.

## How It Works

### 1. Display Detection

The system checks for X11/desktop availability using multiple methods:

- **Environment Variable Check**: Verifies if `$DISPLAY` is set
- **X11 Connection Test**: Attempts to initialize a minimal tkinter window
- **XSet Test** (in shell scripts): Uses `xset q` to test X11 connection

### 2. Automatic Fallback Logic

```
Start Application
       ↓
   Check DISPLAY
       ↓
   DISPLAY set? ──No──→ Launch Streamlit Web UI
       ↓ Yes
   Try GUI Launch
       ↓
   GUI Success? ──No──→ Launch Streamlit Web UI  
       ↓ Yes
   Continue with GUI
```

## Launch Methods

### 1. Main Python Script (`pray.py`)

```bash
python pray.py    # Auto-detects and chooses appropriate UI
```

**Behavior:**
- Checks `$DISPLAY` environment variable
- Tests X11 connection with tkinter
- Falls back to Streamlit if GUI fails

### 2. Enhanced Launch Script (`launch_nemesis.sh`)

```bash
./launch_nemesis.sh           # Interactive menu with auto mode
./launch_nemesis.sh --auto    # Direct auto-detection
```

**Features:**
- Auto mode (option 0) - Default behavior
- Manual selection for specific UI types
- Display availability feedback
- Proper error handling

### 3. Desktop Integration (`nemesis-nexus.desktop`)

Double-click desktop launcher automatically:
- Attempts GUI first
- Falls back to Streamlit web UI
- Opens terminal for user interaction

## Streamlit Web UI Features

When fallback occurs, the system:

1. **Starts Streamlit server** on port 8501
2. **Displays access URLs**:
   - Local: `http://localhost:8501`
   - Network: `http://[local-ip]:8501`
3. **Enables headless mode** (no browser auto-open)
4. **Shows security warnings** for authorized testing only

## Testing the System

### Test Script
```bash
./test_auto_fallback.sh
```

### Manual Testing

**Normal Environment:**
```bash
echo $DISPLAY                    # Should show :0 or similar
./launch_nemesis.sh --auto       # Should try GUI first
```

**Headless Environment:**
```bash
DISPLAY='' ./launch_nemesis.sh --auto    # Should go straight to Streamlit
```

**Force Streamlit:**
```bash
./launch_nemesis.sh              # Select option 3
python pray.py --streamlit       # Direct Streamlit launch
```

## Configuration

### Environment Variables
- `DISPLAY` - X11 display server location
- `STREAMLIT_SERVER_PORT` - Custom port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Custom address (default: 0.0.0.0)

### Shell Functions
- `check_display()` - Test X11 availability
- `start_streamlit()` - Launch web UI with proper URLs
- `fallback_to_streamlit()` - Python fallback function

## Use Cases

### 1. Local Development
- **GUI Available**: Full desktop experience with tkinter interface
- **GUI Unavailable**: Automatic web interface in browser

### 2. Remote/SSH Access
- **SSH with X-Forwarding**: GUI works normally
- **SSH without X-Forwarding**: Auto-falls back to web UI
- **Access via browser**: Use network URL from any device

### 3. Server Deployment
- **Headless Server**: Streamlit web interface only
- **Container Deployment**: No X11, automatic web UI
- **Cloud Instance**: Web interface accessible from anywhere

### 4. CI/CD Integration
- **Automated Testing**: Headless mode compatible
- **Build Systems**: No GUI dependencies
- **Docker Containers**: Web interface only

## Troubleshooting

### Common Issues

**1. GUI Fails to Start**
```
Error: tkinter.TclError: no display name and no $DISPLAY environment variable
```
**Solution**: System will automatically fallback to Streamlit

**2. Streamlit Not Found**
```
Error: streamlit command not found
```
**Solution**: Install Streamlit: `pip install streamlit`

**3. Port Already in Use**
```
Error: Port 8501 is already in use
```
**Solution**: Kill existing process or use different port

### Debug Commands

```bash
# Check display status
echo "DISPLAY=$DISPLAY"
xset q

# Test Python GUI capability
python3 -c "import tkinter; tkinter.Tk().withdraw()"

# Check Streamlit availability
which streamlit
streamlit --version

# Test port availability
lsof -i :8501
```

## Security Considerations

- Web UI accessible over network (intentional for remote access)
- Authentication handled by application layer
- HTTPS not enabled by default (add reverse proxy for production)
- Warning messages displayed for authorized testing only

## Future Enhancements

- [ ] Automatic HTTPS certificate generation
- [ ] Built-in authentication for web interface
- [ ] VPN/tunnel integration for secure remote access
- [ ] Mobile-responsive web interface improvements
- [ ] WebRTC for real-time collaboration features
