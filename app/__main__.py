from app.main import main

# Guarded, because the dispersion study's worker processes are spawned from
# this module on Windows and import it again: without the guard every worker
# would open its own copy of the application.
if __name__ == "__main__":
    raise SystemExit(main())
