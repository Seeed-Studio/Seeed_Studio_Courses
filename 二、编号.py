import re

def parse_markdown(input_file):
    """
    读取 markdown 文件并提取其中的标题。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.readlines()

    # 用于存储标题及其行号
    headers = []
    
    for line_num, line in enumerate(content):
        # 匹配所有标题行，标题以 # 开头
        match = re.match(r'^(#+)\s+(.*)', line.strip())
        if match:
            level = len(match.group(1))  # 获取标题级别（#的数量）
            text = match.group(2)  # 获取标题文本
            headers.append((level, text, line_num, line.strip()))

    return content, headers

def add_numbering_to_headers(headers):
    """
    为标题添加编号，最多支持四级标题。
    如果标题是五级及以上，将不进行编号。
    """
    numbered_headers = []
    counter = [0] * 3  # 用三个计数器用于最多三级标题编号
    
    for level, text, line_num, original_line in headers:
        if level == 1:
            # 一级标题不编号
            numbered_text = original_line
        elif level == 2:
            # 二级标题：判断是否是“课程简介”或“课程目标”
            if text == "课程简介" or text == "课程目标":
                # 不编号
                numbered_text = original_line
            else:
                # 其他二级标题编号为 1.
                counter[0] += 1
                counter[1] = 0  # 三级标题计数器重置
                counter[2] = 0  # 四级标题计数器重置
                numbering = f"{counter[0]}."  # 加上点
                numbered_text = f"## {numbering} {text}"
        elif level == 3:
            # 三级标题编号为 1.1.
            counter[1] += 1
            counter[2] = 0  # 四级标题计数器重置
            numbering = f"{counter[0]}.{counter[1]}."
            numbered_text = f"### {numbering} {text}"
        elif level == 4:
            # 四级标题编号为 1.1.1.
            counter[2] += 1
            numbering = f"{counter[0]}.{counter[1]}.{counter[2]}."
            numbered_text = f"#### {numbering} {text}"
        else:
            # 五级及以上标题不编号
            numbered_text = original_line

        numbered_headers.append((line_num, numbered_text))
    
    return numbered_headers

def create_numbered_markdown(content, numbered_headers):
    """
    根据编号后的标题生成新的 markdown 内容。
    """
    numbered_content = []
    header_idx = 0
    
    for line_num, line in enumerate(content):
        if header_idx < len(numbered_headers) and line_num == numbered_headers[header_idx][0]:
            # 用带编号的标题替换原来的标题
            numbered_content.append(f"{numbered_headers[header_idx][1]}\n")
            header_idx += 1
        else:
            numbered_content.append(line)
    
    return ''.join(numbered_content)

def main(input_file, output_file):
    content, headers = parse_markdown(input_file)
    numbered_headers = add_numbering_to_headers(headers)
    numbered_markdown = create_numbered_markdown(content, numbered_headers)
    
    # 写入新的 markdown 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(numbered_markdown)
    print(f"Numbered markdown content has been written to {output_file}")

if __name__ == "__main__":
    input_file = 'D:/document/GitHub Desktop/courses/Seeed_Studio_Courses/docs/cn/3/17/README_output.md'  # 输入的 markdown 文件路径
    output_file = 'D:/document/GitHub Desktop/courses/Seeed_Studio_Courses/docs/cn/3/17/README.md'  # 输出的 markdown 文件路径
    main(input_file, output_file)
