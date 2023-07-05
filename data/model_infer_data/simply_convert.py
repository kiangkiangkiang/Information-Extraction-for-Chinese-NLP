import json

if __name__ == "__main__":
    result = []
    with open("./verdict8000.json", "r", encoding="utf8") as f:
        for i in f:
            all_content = json.loads(i)
            for content in all_content:
                result.append(content["jfull_compress"])

    with open("./verdict8000.txt", "w", encoding="utf8") as f:
        for i in result:
            f.write(i)
            f.write("\n")
