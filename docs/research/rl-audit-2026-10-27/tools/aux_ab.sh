#!/bin/bash
SP=$(cd $(dirname $0) && pwd)
NAME=audit_auxon_s7 GPU=1 TOTAL=98304 bash $SP/run_train.sh > $SP/audit_auxon.log 2>&1
NAME=audit_auxoff_s7 GPU=1 TOTAL=98304 bash $SP/run_train.sh --aux-b8 0 --aux-near-car 0 > $SP/audit_auxoff.log 2>&1
echo AB_DONE
