import argparse
import json
import sys

from utils.database import Database


def fetch_chat_history(user_id: str, limit_sessions: int) -> dict:
    """Helper to instantiate DB and fetch chat history."""
    database = Database()
    return database.get_full_chat_history(user_id=user_id, limit_sessions=limit_sessions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tải toàn bộ chat history cho một user cụ thể."
    )
    parser.add_argument(
        "--user-id",
        default="6927c763d657a363e82a18ee",
        help="ID của user cần lấy chat history (mặc định: 6927c763d657a363e82a18ee)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Số lượng chat session tối đa cần lấy (mặc định: 50)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="In JSON với định dạng dễ đọc (indent=2).",
    )
    args = parser.parse_args()

    try:
        history = fetch_chat_history(args.user_id, args.limit)
    except Exception as exc:
        print(f"Lỗi khi lấy chat history: {exc}", file=sys.stderr)
        sys.exit(1)

    json_text = json.dumps(
        history,
        indent=2 if args.pretty else None,
        ensure_ascii=False,
    )
    print(json_text)


if __name__ == "__main__":
    main()

