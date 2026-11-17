#!/bin/bash
# Cài bộ dịch/chạy cho MultiPL-E (code_eval_ml.py): Node + TypeScript, Java (javatuples), .NET 8, PHP, Go, Bash — Ubuntu 24.04 (NGC).
# Xong chạy selftest (đáp án mẫu mỗi ngôn ngữ phải QUA trong sandbox sbx). In MLE_XONG.
#   bash scripts/setup_multipl_e.sh > /workspace/setup_mle.log 2>&1
cd /workspace/nn-aivie
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nodejs npm default-jdk-headless php-cli golang-go dotnet-sdk-8.0 > /dev/null || exit 1
npm install -g -s typescript @types/node || exit 1
mkdir -p /opt/c2b/gocache && chmod 777 /opt/c2b /opt/c2b/gocache
[ -s /opt/c2b/javatuples-1.2.jar ] || curl -sL -o /opt/c2b/javatuples-1.2.jar https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar
chmod 644 /opt/c2b/javatuples-1.2.jar
node --version; tsc --version; java -version 2>&1 | head -1; php --version | head -1; go version; dotnet --version
AIVIE_SANDBOX_USER=sbx PYTHONPATH=scripts:src /workspace/evalenv/bin/python -u scripts/code_eval_ml.py selftest | tee /workspace/mle_selftest.log
grep -q "### ml selftest: OK" /workspace/mle_selftest.log || exit 1
echo MLE_XONG
