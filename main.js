const { app, BrowserWindow, session, protocol } = require('electron');
const path = require('path');
const fs = require('fs');

// Must be called before app is ready — grants fetch() + CORS support to custom scheme
protocol.registerSchemesAsPrivileged([
  { scheme: 'mediapipe', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } }
]);

app.whenReady().then(() => {
  // Serve @mediapipe/hands files so WASM, tflite, binarypb all load via fetch()
  protocol.registerBufferProtocol('mediapipe', (request, callback) => {
    const file = decodeURIComponent(request.url.replace('mediapipe://', '').split('?')[0]);
    const filePath = path.join(__dirname, 'node_modules/@mediapipe/hands', file);
    try {
      const data = fs.readFileSync(filePath);
      const ext = path.extname(filePath);
      const mimeType = ext === '.wasm'    ? 'application/wasm'
                     : ext === '.js'      ? 'application/javascript'
                     : ext === '.tflite'  ? 'application/octet-stream'
                     : ext === '.binarypb'? 'application/octet-stream'
                     : 'application/octet-stream';
      callback({ mimeType, data });
    } catch (e) {
      callback({ error: -6 });
    }
  });

  createWindow();
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    frame: false,
    backgroundColor: '#0d0d0f',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
    },
  });

  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') return callback(true);
    callback(false);
  });

  win.loadFile('index.html');
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
