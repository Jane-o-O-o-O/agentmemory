#!/bin/bash
# Email notification after dev run
PROJECT="$1"
STATUS="$2"
DETAILS="$3"
NOW=$(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M')
RESEND_KEY='re_N7rj7411_HFpMGnqoZdQYG2RMvPke4bBA'

SUBJECT="🚀 [${PROJECT}] 开发进度 - ${NOW} (北京时间)"
BODY="项目: ${PROJECT}
时间: ${NOW} (北京时间)
状态: ${STATUS}

${DETAILS}

—— Hermes Agent 自动通知"

# Escape for JSON
SUBJECT_JSON=$(echo "$SUBJECT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))")
BODY_JSON=$(echo "$BODY" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))")

curl -s -X POST 'https://api.resend.com/emails' \
  -H "Authorization: Bearer ${RESEND_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{\"from\": \"Hermes Agent <onboarding@resend.dev>\",\"to\": [\"2689124001@qq.com\"],\"subject\": ${SUBJECT_JSON},\"text\": ${BODY_JSON}}" > /dev/null 2>&1 && echo '✅ 邮件已发送' || echo '❌ 邮件发送失败'
