# Update Config 
def update_config(file_name, sub, output_num=1):
    with open(file_name) as f:
        data_lines = [lines.strip() for lines in list(f)]

    # 文字列置換
    for n, i in enumerate(data_lines):
        if "num_inputs " in i:
            data_lines[n]="num_inputs              = "+str(len(sub.input_coordinates[0]) + len(sub.output_coordinates[0]) + 1)
        if "num_outputs" in i:
            data_lines[n]="num_outputs             = "+str(output_num)

    # 同じファイル名で保存
    with open(file_name, mode="w") as f:
        for i in data_lines:
            f.write("%s\n" % i)