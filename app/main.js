const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1600,
        height: 900,
        title: "ArbItro",
        backgroundColor: '#050A14',
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            webSecurity: false
        }
    });
    mainWindow.loadFile('client/index.html');
}

app.whenReady().then(createWindow);

ipcMain.handle('open-file-dialog', async () => {
    const { filePaths } = await dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections'],
        filters: [{ name: 'Movies', extensions: ['mp4', 'mkv', 'avi'] }]
    });
    return filePaths;
});