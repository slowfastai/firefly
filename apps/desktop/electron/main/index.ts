import { app, BrowserWindow, shell, ipcMain } from 'electron'
import { spawn, ChildProcess } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { update } from './update'

const require = createRequire(import.meta.url)
const __dirname = path.dirname(fileURLToPath(import.meta.url))

// The built directory structure
//
// ├─┬ dist-electron
// │ ├─┬ main
// │ │ └── index.js    > Electron-Main
// │ └─┬ preload
// │   └── index.mjs   > Preload-Scripts
// ├─┬ dist
// │ └── index.html    > Electron-Renderer
//
process.env.APP_ROOT = path.join(__dirname, '../..')

export const MAIN_DIST = path.join(process.env.APP_ROOT, 'dist-electron')
export const RENDERER_DIST = path.join(process.env.APP_ROOT, 'dist')
export const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, 'public')
  : RENDERER_DIST

// Disable GPU Acceleration for Windows 7
if (os.release().startsWith('6.1')) app.disableHardwareAcceleration()

// Set application name for Windows 10+ notifications
if (process.platform === 'win32') app.setAppUserModelId(app.getName())

if (!app.requestSingleInstanceLock()) {
  app.quit()
  process.exit(0)
}

let win: BrowserWindow | null = null
const preload = path.join(__dirname, '../preload/index.mjs')
const indexHtml = path.join(RENDERER_DIST, 'index.html')

// Track running sessions so we can cancel
const sessions = new Map<string, ChildProcess>()

// Resolve a Python executable that respects conda/venv when present
function resolvePythonExecutable(): string {
  const isWin = process.platform === 'win32'
  const bin = (p: string) => path.join(p, isWin ? 'python.exe' : path.join('bin', 'python'))

  const candidates = [
    process.env.PYTHON_EXECUTABLE,
    process.env.PYTHON,
    process.env.CONDA_PREFIX ? bin(process.env.CONDA_PREFIX) : undefined,
    process.env.VIRTUAL_ENV ? bin(process.env.VIRTUAL_ENV) : undefined,
  ].filter(Boolean) as string[]

  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c
    } catch {}
  }

  // Fall back to PATH lookups
  return isWin ? 'python' : 'python3'
}

async function createWindow() {
  win = new BrowserWindow({
    title: 'Main window',
    icon: path.join(process.env.VITE_PUBLIC, 'favicon.ico'),
    webPreferences: {
      preload,
      // Warning: Enable nodeIntegration and disable contextIsolation is not secure in production
      // nodeIntegration: true,

      // Consider using contextBridge.exposeInMainWorld
      // Read more on https://www.electronjs.org/docs/latest/tutorial/context-isolation
      // contextIsolation: false,
    },
  })

  if (VITE_DEV_SERVER_URL) { // #298
    win.loadURL(VITE_DEV_SERVER_URL)
    // Open devTool if the app is not packaged
    win.webContents.openDevTools()
  } else {
    win.loadFile(indexHtml)
  }

  // Test actively push message to the Electron-Renderer
  win.webContents.on('did-finish-load', () => {
    win?.webContents.send('main-process-message', new Date().toLocaleString())
  })

  // Make all links open with the browser, not with the application
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https:')) shell.openExternal(url)
    return { action: 'deny' }
  })

  // Auto update
  update(win)
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  win = null
  if (process.platform !== 'darwin') app.quit()
})

app.on('second-instance', () => {
  if (win) {
    // Focus on the main window if the user tried to open another
    if (win.isMinimized()) win.restore()
    win.focus()
  }
})

app.on('activate', () => {
  const allWindows = BrowserWindow.getAllWindows()
  if (allWindows.length) {
    allWindows[0].focus()
  } else {
    createWindow()
  }
})

// New window example arg: new windows url
ipcMain.handle('open-win', (_, arg) => {
  const childWindow = new BrowserWindow({
    webPreferences: {
      preload,
      nodeIntegration: true,
      contextIsolation: false,
    },
  })

  if (VITE_DEV_SERVER_URL) {
    childWindow.loadURL(`${VITE_DEV_SERVER_URL}#${arg}`)
  } else {
    childWindow.loadFile(indexHtml, { hash: arg })
  }
})

ipcMain.handle('submit-query', async (evt, payload: { engine: string; model?: string; query: string; sessionId?: string }) => {
  const trimmed = (payload?.query ?? '').toString().trim()
  if (!trimmed) return

  const { engine, model } = payload

  const appRoot = process.env.APP_ROOT ?? path.join(__dirname, '../..')
  const scriptPath = path.join(appRoot, 'scripts', 'handle_query.py')
  const sessionId = payload.sessionId || randomUUID()
  const pythonPayload = {
    engine,
    model,
    query: trimmed,
    session_id: sessionId,
  }

  const finalResult = await new Promise<{ sessionId: string; result?: any }>((resolve, reject) => {
    let buffer = ''
    let stderr = ''

    // per-session cancel flag file for cooperative cancel
    const cancelDir = path.join(appRoot, 'cache', 'cancel_flags')
    const cancelFile = path.join(cancelDir, `${sessionId}.flag`)
    try { fs.mkdirSync(cancelDir, { recursive: true }) } catch {}
    try { if (fs.existsSync(cancelFile)) fs.rmSync(cancelFile) } catch {}

    const pythonExe = resolvePythonExecutable()
    const child = spawn(pythonExe, [scriptPath, JSON.stringify({ ...pythonPayload, cancel_file: cancelFile })], {
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: true, // put child in its own process group
      env: { ...process.env, DR_CANCEL_FILE: cancelFile, DR_SESSION_ID: sessionId },
    })

    // store child for cancellation
    sessions.set(sessionId, child)

    child.stdout?.on('data', chunk => {
      buffer += chunk.toString()
      let idx: number
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 1)
        if (!line) continue
        try {
          const evtObj = JSON.parse(line)
          // Forward event to renderer
          evt.sender.send('reasoning-event', { sessionId, ...evtObj })
          // Capture final result for resolve
          if (evtObj?.type === 'final') {
            resolve({ sessionId, result: evtObj.payload })
          }
        } catch (e) {
          // Non-JSON line; forward as log
          evt.sender.send('reasoning-event', { sessionId, type: 'log', payload: { line } })
        }
      }
    })

    child.stderr?.on('data', chunk => {
      stderr += chunk.toString()
    })

    child.on('error', reject)
    child.on('close', (code, signal) => {
      // cleanup session map
      sessions.delete(sessionId)
      try { if (fs.existsSync(cancelFile)) fs.rmSync(cancelFile) } catch {}
      if (code === 0 || !!signal) {
        // If no explicit final event, still resolve with sessionId
        resolve({ sessionId })
      } else {
        const message = stderr.trim() || `Python script exited with code ${code}`
        reject(new Error(message))
      }
    })
  })

  return finalResult
})

// Forward clarification answer from renderer to the Python child via stdin
ipcMain.handle('send-clarification', async (evt, payload: { sessionId: string; text: string }) => {
  const { sessionId, text } = payload || ({} as any)
  if (!sessionId || typeof text !== 'string') return { ok: false, reason: 'bad_request' }
  const child = sessions.get(sessionId)
  if (!child) return { ok: false, reason: 'not_found' }
  try {
    const msg = JSON.stringify({ type: 'clarification_answer', payload: { text } }) + '\n'
    child.stdin?.write(msg)
    return { ok: true }
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : 'write_failed' }
  }
})

ipcMain.handle('cancel-session', async (evt, payload: { sessionId: string }) => {
  const { sessionId } = payload || ({} as any)
  if (!sessionId) return { ok: false, reason: 'missing sessionId' }
  const child = sessions.get(sessionId)
  if (!child) return { ok: false, reason: 'not_found' }

  try {
    // Send cooperative cancel message via stdin first
    try {
      child.stdin?.write(JSON.stringify({ type: 'cancel', session_id: sessionId }) + '\n')
      child.stdin?.end()
    } catch {}

    // Also flip the cancel flag file if provided
    try {
      const cancelFile = (child.spawnargs?.find?.(() => false), process.env.DR_CANCEL_FILE) // backup if accessible
      // best-effort: reproduce expected path
      const appRoot = process.env.APP_ROOT || path.join(__dirname, '../..')
      const cf = path.join(appRoot, 'cache', 'cancel_flags', `${sessionId}.flag`)
      fs.mkdirSync(path.dirname(cf), { recursive: true })
      fs.writeFileSync(cf, '1')
    } catch {}

    // Then try graceful stop for entire process group
    try {
      process.kill(-(child.pid as number), 'SIGTERM')
    } catch {
      child.kill('SIGTERM')
    }

    const timer = setTimeout(() => {
      try {
        process.kill(-(child.pid as number), 'SIGKILL')
      } catch {
        try { child.kill('SIGKILL') } catch {}
      }
    }, 1500)

    child.once('exit', () => {
      clearTimeout(timer)
      evt.sender.send('reasoning-event', { sessionId, type: 'cancelled', payload: { message: 'Cancelled by user' } })
    })
    return { ok: true }
  } catch (e: any) {
    return { ok: false, reason: e?.message || 'kill_failed' }
  } finally {
    // Remove from map immediately; 'exit' handler also deletes on close from submit handler
    sessions.delete(sessionId)
  }
})
