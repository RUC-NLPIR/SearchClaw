#!/bin/bash
#
# Setup Chrome to always launch with CDP (Chrome DevTools Protocol) enabled.
#
# This creates a macOS app wrapper "Chrome CDP.app" that launches your real
# Chrome with --remote-debugging-port=9222, using your existing profile
# (all bookmarks, cookies, extensions, logins preserved).
#
# After setup, use "Chrome CDP" from Spotlight/Dock instead of "Google Chrome".
# The search agent can then connect via CDP to your real browser session.
#
# Usage:
#   ./scripts/setup_chrome_cdp.sh          # Create the app (default port 9222)
#   ./scripts/setup_chrome_cdp.sh 9333     # Use a custom port
#
set -euo pipefail

CDP_PORT="${1:-9222}"
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
APP_NAME="Chrome CDP"
APP_DIR="$HOME/Applications/${APP_NAME}.app"
SCRIPT_DIR="${APP_DIR}/Contents/MacOS"
PLIST_DIR="${APP_DIR}/Contents"
ICON_SOURCE="/Applications/Google Chrome.app/Contents/Resources/app.icns"

if [ ! -f "$CHROME_PATH" ]; then
    echo "Error: Chrome not found at $CHROME_PATH"
    exit 1
fi

echo "Creating ${APP_NAME}.app with CDP on port ${CDP_PORT}..."
echo ""

# Create app bundle structure
mkdir -p "$SCRIPT_DIR"
mkdir -p "${APP_DIR}/Contents/Resources"

# Copy Chrome icon
if [ -f "$ICON_SOURCE" ]; then
    cp "$ICON_SOURCE" "${APP_DIR}/Contents/Resources/app.icns"
fi

# Create the launcher script
cat > "${SCRIPT_DIR}/${APP_NAME}" << 'LAUNCHER'
#!/bin/bash
# Chrome CDP Launcher — opens Chrome with DevTools Protocol enabled.
# Uses your real Chrome profile so all cookies/extensions/logins work.

CDP_PORT="__CDP_PORT__"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Use a dedicated user-data-dir (required for Chrome 146+).
# We copy from the real profile on first run so you keep all your data.
REAL_PROFILE="$HOME/Library/Application Support/Google/Chrome"
CDP_PROFILE="$HOME/Library/Application Support/Google/Chrome-CDP"

# First run: copy the real profile
if [ ! -d "$CDP_PROFILE" ]; then
    echo "First run — copying Chrome profile to $CDP_PROFILE ..."
    echo "This may take a minute. Your original profile is not modified."
    cp -R "$REAL_PROFILE" "$CDP_PROFILE" 2>/dev/null || {
        echo "Warning: Could not copy profile. Creating fresh profile."
        mkdir -p "$CDP_PROFILE"
    }
fi

exec "$CHROME" \
    --remote-debugging-port="$CDP_PORT" \
    --user-data-dir="$CDP_PROFILE" \
    --no-first-run \
    --no-default-browser-check \
    --disable-session-crashed-bubble \
    --hide-crash-restore-bubble \
    "$@"
LAUNCHER

# Substitute the actual port
sed -i '' "s/__CDP_PORT__/${CDP_PORT}/" "${SCRIPT_DIR}/${APP_NAME}"
chmod +x "${SCRIPT_DIR}/${APP_NAME}"

# Create Info.plist
cat > "${PLIST_DIR}/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>com.search-agent.chrome-cdp</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleIconFile</key>
    <string>app.icns</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

echo "Done! Created: ${APP_DIR}"
echo ""
echo "How to use:"
echo "  1. Quit Google Chrome completely"
echo "  2. Open '${APP_NAME}' from Spotlight, Finder, or:"
echo "     open '$APP_DIR'"
echo "  3. Chrome opens normally — use it as you always do"
echo "  4. The search agent connects via CDP on port ${CDP_PORT}"
echo ""
echo "First launch copies your Chrome profile to:"
echo "  ~/Library/Application Support/Google/Chrome-CDP"
echo "This is a separate copy — your original profile is never modified."
echo ""
echo "To verify CDP is working:"
echo "  curl http://127.0.0.1:${CDP_PORT}/json/version"
echo ""
echo "Tip: Drag '${APP_NAME}' to your Dock for easy access."
echo "     You can find it at: ${APP_DIR}"
