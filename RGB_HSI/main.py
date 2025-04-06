

# 使用 with 语句时，文件会在上下文结束时自动关闭，无需显式调用 close()
def read_file_and_write(source_file, target_file):
    with open(source_file, 'r') as file:
        finalList = []
        # for就是在迭代文件中的每一行内容  读取每一行
        for line in file:
            line = line[0:-5]
            print(line)
            finalList.append(line)

    # 写入目标文件
    with open(target_file, "w") as dstFile:
        for item in finalList:
            dstFile.write(item)
            dstFile.write("\n")
read_file_and_write('train1.txt', 'newtrain.txt')
