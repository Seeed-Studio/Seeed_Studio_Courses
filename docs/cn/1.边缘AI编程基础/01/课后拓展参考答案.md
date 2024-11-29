# 课后拓展参考答案

## 思考题参考答案

1. **如果你可以为日常生活中的某个物品添加边缘 AI 功能，你会选择什么？希望它具备什么智能功能？**
   <details>
   <summary>点击查看答案</summary>
   示例答案：为冰箱添加边缘 AI 功能，让它能够识别内部的食品种类和数量，并提供实时过期提醒。冰箱还能根据现有食材推荐菜谱，帮助减少浪费。
   </details>

2. **为什么我们在不同场景下需要用到 `ls` 命令？在实际开发中，这个命令可以帮助我们解决什么问题？**
   <details>
   <summary>点击查看答案</summary>
   示例答案：`ls` 命令可以快速查看当前目录的文件和文件夹，帮助我们确认文件是否存在或验证目录结构。在开发中，它能够帮助检查项目文件是否正确生成，或者定位需要修改的文件。
   </details>

3. **使用 `cd` 命令时，为什么路径管理会影响我们的文件管理效率？有什么技巧能提高效率？**
   <details>
   <summary>点击查看答案</summary>
   示例答案：路径管理直接决定了我们操作文件和目录的效率。频繁使用 `cd` 命令切换目录时，记住相对路径可以减少复杂操作。同时可以使用快捷命令（如 `cd ~` 返回主目录、`cd ..` 返回上一级目录）提高效率。另外，`tab` 键自动补全功能可以显著加快导航速度。
   </details>

---

## 实践任务参考答案

### 终端命令练习

#### 练习 1：目录和文件管理

- 创建一个名为 my_project 的目录，在该目录内创建 README.md 文件，并在其中输入“Hello, Edge AI!”。

   <details>
   <summary>点击查看答案</summary>

   1. 创建目录：

      ```bash
      mkdir my_project
      ```

   2. 创建文件：

      ```bash
      cd my_project
      touch README.md
      ```

   3. 编辑文件：

      ```bash
      echo "Hello, Edge AI!" > README.md
      ```

   4. 查看内容：

      ```bash
      cat README.md
      ```

   </details>

#### 练习 2：文件导航

- 进入 Documents 目录，列出该目录下的所有文件和文件夹。

   <details>
   <summary>点击查看答案</summary>

   1. 切换到 Documents 目录：

      ```bash
      cd ~/Documents
      ```

   2. 列出目录内容：

      ```bash
      ls
      ```

   </details>

#### 练习 3：复制与删除文件

- 创建一个名为 backup.txt 的文件，将其复制为 backup_copy.txt，然后删除原文件。

   <details>
   <summary>点击查看答案</summary>

   1. 创建文件：

      ```bash
      touch backup.txt
      ```

   2. 复制文件：

      ```bash
      cp backup.txt backup_copy.txt
      ```

   3. 删除原文件：

      ```bash
      rm backup.txt
      ```

   4. 验证结果：

      ```bash
      ls
      ```

   </details>
