#/bin/bash

SHELL_FOLDER=$(dirname $(readlink -f "$0"))


${SHELL_FOLDER}/cpplint.py --recursive --quiet ${SHELL_FOLDER}/demo ${SHELL_FOLDER}/node ${SHELL_FOLDER}/core ${SHELL_FOLDER}/common