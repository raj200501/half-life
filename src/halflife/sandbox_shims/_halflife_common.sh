# Shared helpers for the half-life sandbox shims. Sourced by each shim, never run.
# The shims stand in for networked tools inside the sandbox: a network use is
# logged as one JSON line {"target": ..., "via": <tool>} to $HALFLIFE_NETLOG and
# the shim exits non-zero without touching the network. Offline uses (version,
# help, and local subcommands) pass through to the real tool.

hl_json() {
    printf '%s' "$1" | tr '\n\r\t' '   ' | tr -d '\000-\037' |
        sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | cut -c1-500
}

hl_log() {
    # hl_log VIA TARGET
    [ -n "${HALFLIFE_NETLOG:-}" ] || return 0
    printf '{"target": "%s", "via": "%s"}\n' "$(hl_json "$2")" "$(hl_json "$1")" \
        >>"$HALFLIFE_NETLOG" 2>/dev/null || true
}

hl_real() {
    # hl_real NAME: print the first NAME on PATH that is not in the shim directory.
    hl_self=$(cd "$hl_here" && pwd -P)
    hl_old_ifs=$IFS
    IFS=:
    set -f
    for hl_dir in $PATH; do
        [ -n "$hl_dir" ] || continue
        hl_real_dir=$(cd "$hl_dir" 2>/dev/null && pwd -P) || continue
        [ "$hl_real_dir" = "$hl_self" ] && continue
        if [ -x "$hl_dir/$1" ] && [ ! -d "$hl_dir/$1" ]; then
            IFS=$hl_old_ifs
            set +f
            printf '%s\n' "$hl_dir/$1"
            return 0
        fi
    done
    IFS=$hl_old_ifs
    set +f
    return 1
}

hl_passthrough() {
    # hl_passthrough NAME ARGS...: run the real tool, or fail if there is none.
    hl_name=$1
    shift
    if hl_path=$(hl_real "$hl_name"); then
        exec "$hl_path" "$@"
    fi
    echo "$hl_name: command not found" >&2
    exit 127
}

hl_refuse() {
    # hl_refuse VIA TARGET EXIT_CODE
    hl_log "$1" "$2"
    echo "$1: network access is disabled in the half-life sandbox" >&2
    exit "$3"
}

hl_urls() {
    # Arguments that look like URLs, or every argument when none does.
    hl_found=""
    for hl_arg in "$@"; do
        case "$hl_arg" in
            *://*) hl_found="${hl_found:+$hl_found }$hl_arg" ;;
        esac
    done
    if [ -n "$hl_found" ]; then
        printf '%s' "$hl_found"
    else
        printf '%s' "$*"
    fi
}

hl_subcommand() {
    # First argument that does not start with a dash.
    for hl_arg in "$@"; do
        case "$hl_arg" in
            -*) ;;
            *) printf '%s' "$hl_arg"; return 0 ;;
        esac
    done
    return 0
}
