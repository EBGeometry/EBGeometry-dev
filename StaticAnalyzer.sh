#!/usr/bin/env bash

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p logs

# Store results
declare -A RESULTS

run_check() {
    local name="$1"
    shift
    echo -e "${YELLOW}Running $name...${NC}"
    if "$@" >"logs/${name}.log" 2>&1; then
        RESULTS[$name]="PASS"
    else
        RESULTS[$name]="FAIL"
    fi
}

# Run tools
run_check "clang-tidy" clang-tidy --extra-arg=-std=c++20 Source/*.hpp Tests/*.hpp
run_check "clang-format" bash -c 'find Source Exec \( -name "*.hpp" -o -name "*.cpp" \) -exec clang-format --dry-run --Werror {} +'
run_check "codespell" codespell Source Exec
run_check "doxygen" doxygen doxyfile

# Final summary
echo
echo "==================== Summary ===================="
FAILED=0
for tool in "${!RESULTS[@]}"; do
    if [[ "${RESULTS[$tool]}" == "PASS" ]]; then
        echo -e "$tool: ${GREEN}PASS${NC}"
    else
        echo -e "$tool: ${RED}FAIL${NC} (see logs/$tool.log)"
        FAILED=1
    fi
done
echo "================================================="
echo

# Exit code
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All checks PASSED ✅${NC}"
    exit 0
else
    echo -e "${RED}Some checks FAILED ❌ — check logs/ for details.${NC}"
    exit 1
fi
