from api.repositories.items import list_items

if __name__ == "__main__":
    items = list_items()
    print(f"Total items in database: {len(items)}")
    if items:
        print(f"First item: {items[0]}")