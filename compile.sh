#!/bin/bash
# Compile getp and CSSRX on your local machine, required g++ >= 4.8.5
ver=$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)

IFS=. read -r major minor patch <<< "$ver"
patch=${patch:-0}

req_major=4
req_minor=8
req_patch=5

if   (( major > req_major )) \
  || (( major == req_major && minor > req_minor )) \
  || (( major == req_major && minor == req_minor && patch >= req_patch )); then
    echo "OK: g++ >= 4.8.5 (current: $ver)"
else
    echo "ERROR: g++ < 4.8.5 (current: $ver)"
    exit 1
fi

echo "Compiling getp"
cd em3na/bin/src/getp
  make clean
  make -j4 all
  cp getp ../..
cd -

echo "Compiling CSSRX"
cd em3na/bin/src/CSSR
  make clean
  make -j4 all
  cp exe/CSSRX ../..
cd -


