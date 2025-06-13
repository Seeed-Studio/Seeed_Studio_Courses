import re

def replace_html_headers_with_markdown(input_text):
    # 替换 <h1> 标签为 Markdown 格式的 '#'
    input_text = re.sub(r'<h1 id="[^"]*">([^<]+)</h1>', r'# \1', input_text)
    
    # 替换 <h2> 标签为 Markdown 格式的 '##'
    input_text = re.sub(r'<h2 id="[^"]*">([^<]+)</h2>', r'## \1', input_text)
    
    # 替换 <h3> 标签为 Markdown 格式的 '###'
    input_text = re.sub(r'<h3 id="[^"]*">([^<]+)</h3>', r'### \1', input_text)
    
    # 替换 <h4> 标签为 Markdown 格式的 '####'
    input_text = re.sub(r'<h4 id="[^"]*">([^<]+)</h4>', r'#### \1', input_text)
    
    # 替换 <h5> 标签为 Markdown 格式的 '#####'
    input_text = re.sub(r'<h5 id="[^"]*">([^<]+)</h5>', r'##### \1', input_text)
    
    # 替换 <h6> 标签为 Markdown 格式的 '######'
    input_text = re.sub(r'<h6 id="[^"]*">([^<]+)</h6>', r'###### \1', input_text)
    
    return input_text

def process_markdown_file(input_file, output_file):
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # 替换 HTML 标题为 Markdown 标题
    output_text = replace_html_headers_with_markdown(input_text)

    # 写入新的文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(output_text)
    print(f"文件已保存为: {output_file}")

# 示例：指定输入文件和输出文件
input_file = 'D:/document/GitHub Desktop/courses/Seeed_Studio_Courses/docs/cn/3/17/README.md'  # 输入的 Markdown 文件路径
output_file = 'D:/document/GitHub Desktop/courses/Seeed_Studio_Courses/docs/cn/3/17/README_output.md'  # 输出的 Markdown 文件路径

# 处理文件
process_markdown_file(input_file, output_file)
