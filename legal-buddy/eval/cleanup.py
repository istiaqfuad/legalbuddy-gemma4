"""Drop the throwaway eval collections from the remote Qdrant. Safe: only touches
collections whose name starts with 'legal_acts_eval_', never the production ones."""
from common import build_qdrant_client

PREFIX = "legal_acts_eval_"


def main():
    client = build_qdrant_client()
    names = [c.name for c in client.get_collections().collections]
    for name in names:
        if name.startswith(PREFIX):
            client.delete_collection(collection_name=name)
            print(f"dropped {name}")
    print("remaining:", [n for n in
                          (c.name for c in client.get_collections().collections)])


if __name__ == "__main__":
    main()
