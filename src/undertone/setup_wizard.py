"""Interactive setup wizard for Undertone."""

import shutil
import subprocess

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from undertone import config
from undertone import service
from undertone.core import detect_session

console = Console()


def check_system_deps() -> dict:
    """Check if required system dependencies are installed."""
    session = detect_session()
    deps = {}

    if session == "wayland":
        # Clipboard: wl-copy (preferred) or xclip via XWayland
        deps["wl-clipboard"] = shutil.which("wl-copy") is not None
        # Key simulation: wtype (wlroots) or ydotool (all) or xdotool (XWayland)
        has_key_tool = (
            shutil.which("wtype") is not None
            or shutil.which("ydotool") is not None
            or shutil.which("xdotool") is not None
        )
        deps["key-sim (wtype/ydotool/xdotool)"] = has_key_tool
    else:
        deps["xclip"] = shutil.which("xclip") is not None
        deps["xdotool"] = shutil.which("xdotool") is not None

    # Check audio
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        deps["audio"] = any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        deps["audio"] = False

    return deps


def install_missing_deps(missing: list) -> bool:
    """Attempt to install missing dependencies."""
    console.print("\n[yellow]trying to install missing deps...[/yellow]")

    apt_packages = []
    for dep in missing:
        if dep == "xclip":
            apt_packages.append("xclip")
        elif dep == "xdotool":
            apt_packages.append("xdotool")
        elif dep == "wl-clipboard":
            apt_packages.append("wl-clipboard")
        elif "key-sim" in dep:
            # Install wtype (works on wlroots) and ydotool (works everywhere)
            apt_packages.append("wtype")
            apt_packages.append("ydotool")

    if apt_packages:
        try:
            cmd = ["sudo", "apt", "install", "-y"] + apt_packages
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                console.print("[red]failed to install. try manually:[/red]")
                console.print(f"[dim]sudo apt install {' '.join(apt_packages)}[/dim]")
                return False
        except Exception:
            console.print("[red]couldn't run apt. install manually:[/red]")
            console.print(f"[dim]sudo apt install {' '.join(apt_packages)}[/dim]")
            return False

    return True


def validate_api_key(api_key: str) -> tuple:
    """Validate the Groq API key by making a test request."""
    try:
        resp = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return True, None
        elif resp.status_code == 401:
            return False, "invalid_api_key"
        else:
            return False, f"api_error_{resp.status_code}"
    except httpx.TimeoutException:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def run_setup():
    """Run the interactive setup wizard."""
    console.print()
    console.print("[bold cyan]aight let's get you set up real quick[/bold cyan]")
    console.print()
    console.print("[dim]" + "-" * 40 + "[/dim]")
    console.print()

    # Detect session
    session = detect_session()
    console.print(f"[bold]display server:[/bold] [cyan]{session}[/cyan]")
    console.print()

    # Check system deps
    console.print("[bold]checking system deps...[/bold]")
    deps = check_system_deps()

    all_good = True
    for dep, found in deps.items():
        if found:
            console.print(f"   [green]|- {dep}: found[/green]")
        else:
            console.print(f"   [red]|- {dep}: missing[/red]")
            all_good = False

    if not all_good:
        missing = [d for d, found in deps.items() if not found]
        if "audio" in missing:
            console.print("\n[red]no audio input detected. check your mic![/red]")
            missing.remove("audio")

        if missing:
            if Confirm.ask("\n[yellow]want me to try installing missing deps?[/yellow]"):
                if not install_missing_deps(missing):
                    return False
                # Recheck
                deps = check_system_deps()
                if not all(deps.values()):
                    console.print("\n[red]still missing deps. fix and try again[/red]")
                    return False
            else:
                console.print("\n[yellow]install them manually and run /setup again[/yellow]")
                return False

    console.print()

    # Get API key
    console.print("[bold]drop your Groq API key[/bold]")
    console.print("[dim](get one free at console.groq.com/keys)[/dim]")
    console.print()

    while True:
        api_key = Prompt.ask("   [cyan]key[/cyan]", password=True)

        if not api_key:
            console.print("   [red]need a key to continue[/red]")
            continue

        if not api_key.startswith("gsk_"):
            console.print("   [yellow]hmm that doesn't look like a groq key[/yellow]")
            console.print("   [dim]should start with gsk_[/dim]")
            continue

        console.print("   [dim]validating...[/dim]", end=" ")
        valid, error = validate_api_key(api_key)

        if valid:
            console.print("[green]we're in[/green]")
            break
        else:
            console.print(f"[red]nah that ain't it[/red]")
            if error == "invalid_api_key":
                console.print("   [dim]double check your key and try again[/dim]")
            else:
                console.print(f"   [dim]error: {error}[/dim]")

    # Save API key
    config.save_api_key(api_key)
    console.print()

    # Hotkey customization
    if Confirm.ask("[bold]wanna customize hotkeys?[/bold]", default=False):
        console.print()
        console.print("[dim]examples: Key.ctrl_r, Key.f8, Key.alt_l[/dim]")

        current_ptt, current_toggle = config.get_hotkeys()

        ptt = Prompt.ask(
            f"   [cyan]push-to-talk[/cyan] [dim](default: {current_ptt})[/dim]",
            default=current_ptt,
        )
        toggle = Prompt.ask(
            f"   [cyan]toggle mode[/cyan] [dim](default: {current_toggle})[/dim]",
            default=current_toggle,
        )

        config.set_hotkeys(ptt, toggle)
        console.print("   [green]locked in[/green]")
    else:
        # Save default config
        cfg = config.load_config()
        config.save_config(cfg)

    console.print()

    # Install service
    console.print("[bold]installing service...[/bold]")
    if service.install_service():
        console.print("   [green]'- auto-start on login: enabled[/green]")
    else:
        console.print("   [red]'- failed to install service[/red]")
        return False

    # Start service
    console.print()
    console.print("[dim]starting undertone...[/dim]")
    if service.start_service():
        console.print("[green]service started[/green]")
    else:
        console.print("[red]failed to start service[/red]")
        return False

    console.print()
    console.print("[dim]" + "-" * 40 + "[/dim]")
    console.print()

    # Success message
    ptt, toggle = config.get_hotkeys()
    ptt_display = ptt.replace("Key.", "").replace("_", " ").title()
    toggle_display = toggle.replace("Key.", "").upper()

    console.print("[bold green]you're all set fam![/bold green]")
    console.print()
    console.print(f"   [cyan]hold {ptt_display}[/cyan] = push-to-talk")
    console.print(f"   [cyan]tap {toggle_display}[/cyan] = toggle on/off")
    console.print()
    console.print("[dim]undertone is now running in the background[/dim]")
    console.print("[dim]just start talking wherever you type[/dim]")
    console.print()

    return True
