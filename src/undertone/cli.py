"""Interactive CLI for Undertone."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from undertone import __version__
from undertone import config
from undertone import service
from undertone.setup_wizard import run_setup

console = Console()


def print_banner():
    """Print the undertone banner."""
    banner = """
  [bold cyan]╭────────────────────────────────────────╮[/bold cyan]
  [bold cyan]│[/bold cyan]  [bold white]UNDERTONE[/bold white]                            [bold cyan]│[/bold cyan]
  [bold cyan]│[/bold cyan]  [dim]talk it. type it. ship it.[/dim]           [bold cyan]│[/bold cyan]
  [bold cyan]╰────────────────────────────────────────╯[/bold cyan]
"""
    console.print(banner)


def get_status_display() -> str:
    """Get the current status as a styled string."""
    if not config.is_configured():
        return "[cyan]waiting for setup[/cyan]"
    elif service.is_running():
        return "[green]vibing[/green]"
    else:
        return "[yellow]sleeping[/yellow]"


def print_status_line():
    """Print the current status line."""
    status = get_status_display()
    console.print(f"  [bold]Status:[/bold] {status}")

    if config.is_configured():
        ptt, toggle = config.get_hotkeys()
        ptt_display = ptt.replace("Key.", "").replace("_", " ").title()
        toggle_display = toggle.replace("Key.", "").upper()
        console.print(f"  [bold]Hotkeys:[/bold] {ptt_display} (hold) | {toggle_display} (toggle)")
    console.print()


def print_commands():
    """Print available commands."""
    console.print("  [dim]Commands:[/dim]")
    console.print("    [cyan]/setup[/cyan]   get your keys in here")
    console.print("    [cyan]/start[/cyan]   let's gooo")
    console.print("    [cyan]/stop[/cyan]    take a break")
    console.print("    [cyan]/status[/cyan]  what's the vibe?")
    console.print("    [cyan]/logs[/cyan]    receipts")
    console.print("    [cyan]/config[/cyan]  tweak the drip")
    console.print("    [cyan]/help[/cyan]    need backup?")
    console.print("    [cyan]/quit[/cyan]    peace out")
    console.print()


def cmd_setup():
    """Run the setup wizard."""
    run_setup()


def cmd_start():
    """Start the service."""
    if not config.is_configured():
        console.print("[yellow]hold up, run /setup first[/yellow]")
        return

    if service.is_running():
        console.print("[yellow]already vibing fam[/yellow]")
        return

    console.print("[dim]starting...[/dim]")
    if service.start_service():
        console.print("[green]lesss gooo! undertone is now listening...[/green]")
        ptt, toggle = config.get_hotkeys()
        ptt_display = ptt.replace("Key.", "").replace("_", " ").title()
        toggle_display = toggle.replace("Key.", "").upper()
        console.print(f"   hold [cyan]{ptt_display}[/cyan] or tap [cyan]{toggle_display}[/cyan] to speak")
    else:
        console.print("[red]couldn't start the service[/red]")
        console.print("[dim]check /logs for more info[/dim]")


def cmd_stop():
    """Stop the service."""
    if not service.is_running():
        console.print("[yellow]already sleeping[/yellow]")
        return

    console.print("[dim]stopping...[/dim]")
    if service.stop_service():
        console.print("[yellow]aight undertone is taking a nap. run /start when you need me[/yellow]")
    else:
        console.print("[red]couldn't stop the service[/red]")


def cmd_status():
    """Show detailed status."""
    console.print()
    console.print("[bold]the vibe check:[/bold]")

    status = service.get_status()

    if status["running"]:
        uptime = status.get("uptime", "unknown")
        console.print(f"   [green]|- service: running for {uptime}[/green]")
    elif status["installed"]:
        console.print("   [yellow]|- service: stopped[/yellow]")
    else:
        console.print("   [red]|- service: not installed[/red]")

    if config.api_key_exists():
        console.print("   [green]|- api key: configured[/green]")
    else:
        console.print("   [red]|- api key: not set[/red]")

    mode = config.get_privacy_mode()
    if mode == "local":
        console.print("   [cyan]'- mode: local (privacy mode)[/cyan]")
    else:
        console.print("   [green]'- mode: cloud (groq)[/green]")

    console.print()


def cmd_logs():
    """Show recent logs."""
    console.print()
    console.print("[bold]receipts:[/bold]")
    console.print()
    logs = service.get_logs(lines=15)
    console.print(f"[dim]{logs}[/dim]")
    console.print()


def cmd_config():
    """Interactive config editor."""
    console.print()
    console.print("[bold]settings:[/bold]")
    console.print()
    console.print("  1. API key")
    console.print("  2. Hotkeys")
    console.print("  3. Privacy mode (cloud/local)")
    console.print("  4. Text cleanup on/off")
    console.print("  5. LLM grammar fix on/off")
    console.print("  6. Back")
    console.print()

    choice = Prompt.ask("[cyan]pick one[/cyan]", choices=["1", "2", "3", "4", "5", "6"], default="6")

    if choice == "1":
        console.print()
        new_key = Prompt.ask("[cyan]new API key[/cyan]", password=True)
        if new_key:
            config.save_api_key(new_key)
            console.print("[green]saved[/green]")
            if service.is_running():
                service.restart_service()
                console.print("[dim]service restarted[/dim]")

    elif choice == "2":
        console.print()
        console.print("[dim]examples: Key.ctrl_r, Key.f8, Key.alt_l[/dim]")
        ptt, toggle = config.get_hotkeys()

        new_ptt = Prompt.ask(f"[cyan]push-to-talk[/cyan]", default=ptt)
        new_toggle = Prompt.ask(f"[cyan]toggle[/cyan]", default=toggle)

        config.set_hotkeys(new_ptt, new_toggle)
        console.print("[green]saved[/green]")

        if service.is_running():
            service.restart_service()
            console.print("[dim]service restarted[/dim]")

    elif choice == "3":
        console.print()
        current = config.get_privacy_mode()
        console.print(f"[dim]currently: {current}[/dim]")
        console.print()
        console.print("  1. [green]cloud[/green] - Groq API, <1s response")
        console.print("  2. [cyan]local[/cyan] - runs on your CPU, ~3-5s response")
        console.print()

        mode_choice = Prompt.ask("[cyan]pick[/cyan]", choices=["1", "2"], default="1" if current == "cloud" else "2")
        new_mode = "cloud" if mode_choice == "1" else "local"
        config.set_privacy_mode(new_mode)

        if new_mode == "local":
            console.print("[cyan]switched to local-only mode[/cyan]")
            console.print("[dim]your voice never leaves your machine[/dim]")
        else:
            console.print("[green]switched to cloud mode[/green]")

        if service.is_running():
            service.restart_service()

    elif choice == "4":
        console.print()
        current = config.get_cleanup_enabled()
        console.print(f"[dim]currently: {'on' if current else 'off'}[/dim]")

        new_state = Confirm.ask("[cyan]enable text cleanup?[/cyan]", default=current)
        config.set_cleanup_enabled(new_state)
        console.print(f"[green]cleanup {'enabled' if new_state else 'disabled'}[/green]")

        if service.is_running():
            service.restart_service()

    elif choice == "5":
        console.print()
        current = config.get_cleanup_llm_enabled()
        console.print(f"[dim]currently: {'on' if current else 'off'}[/dim]")
        console.print("[dim]uses Groq LLM to fix grammar and punctuation[/dim]")

        new_state = Confirm.ask("[cyan]enable LLM grammar fix?[/cyan]", default=current)
        config.set_cleanup_llm_enabled(new_state)
        console.print(f"[green]LLM grammar fix {'enabled' if new_state else 'disabled'}[/green]")

        if service.is_running():
            service.restart_service()

    console.print()


def cmd_help():
    """Show help."""
    console.print()
    console.print("[bold]undertone commands:[/bold]")
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Command", style="cyan")
    table.add_column("Description")

    table.add_row("/setup", "Run the setup wizard (API key, hotkeys, etc.)")
    table.add_row("/start", "Start the voice typing service")
    table.add_row("/stop", "Stop the voice typing service")
    table.add_row("/status", "Show detailed status info")
    table.add_row("/logs", "View recent service logs")
    table.add_row("/config", "Edit settings (hotkeys, privacy mode, etc.)")
    table.add_row("/help", "Show this help message")
    table.add_row("/quit", "Exit CLI (service keeps running)")

    console.print(table)
    console.print()
    console.print("[dim]tip: service runs in background, so you can close this anytime[/dim]")
    console.print()


def run_repl():
    """Run the interactive REPL."""
    commands = {
        "/setup": cmd_setup,
        "/start": cmd_start,
        "/stop": cmd_stop,
        "/status": cmd_status,
        "/logs": cmd_logs,
        "/config": cmd_config,
        "/help": cmd_help,
        "/quit": lambda: None,
        "/exit": lambda: None,
        "/q": lambda: None,
    }

    while True:
        try:
            user_input = Prompt.ask("[bold magenta]undertone>[/bold magenta]").strip().lower()

            if not user_input:
                continue

            if user_input in ("/quit", "/exit", "/q"):
                console.print("[dim]peace out[/dim]")
                break

            if user_input in commands:
                commands[user_input]()
            elif user_input.startswith("/"):
                console.print(f"[red]unknown command: {user_input}[/red]")
                console.print("[dim]type /help for available commands[/dim]")
            else:
                console.print("[dim]commands start with / (try /help)[/dim]")

        except KeyboardInterrupt:
            console.print()
            console.print("[dim]peace out[/dim]")
            break
        except EOFError:
            break


def main():
    """Main entry point."""
    print_banner()
    print_status_line()

    # Auto-run setup if not configured
    if not config.is_configured():
        console.print("[yellow]looks like this is your first time![/yellow]")
        console.print("[dim]let's get you set up...[/dim]")
        console.print()
        run_setup()
        console.print()
    else:
        print_commands()

    run_repl()


if __name__ == "__main__":
    main()
