import json
import numpy as np

def LCS(X, Y):

    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]
    longest_L = [[[]] * (n + 1) for i in range(m + 1)]
    longest = 0
    lcs_set = []

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
                longest_L[i][j] = []
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
                longest_L[i][j] = longest_L[i - 1][j - 1] + [X[i - 1]]
                if L[i][j] > longest:
                    lcs_set = []
                    lcs_set.append(longest_L[i][j])
                    longest = L[i][j]
                elif L[i][j] == longest and longest != 0:
                    lcs_set.append(longest_L[i][j])
            else:
                if L[i - 1][j] > L[i][j - 1]:
                    L[i][j] = L[i - 1][j]
                    longest_L[i][j] = longest_L[i - 1][j]
                else:
                    L[i][j] = L[i][j - 1]
                    longest_L[i][j] = longest_L[i][j - 1]

    if len(lcs_set) > 0:
        return lcs_set[0]
    else:
        return lcs_set

def parse_sketch(sketch):
    sketch_clips = sketch.split(", ")
    return_sketch = []
    for sketch_clip in sketch_clips:
        sketch_clip_split = sketch_clip.split(" ")
        if len(sketch_clip_split) == 1:
            return_sketch.append(sketch_clip)
        elif len(sketch_clip_split) == 3:
            return_sketch.append(" ".join(sketch_clip_split[:-1]))
        else:
            return_sketch.append(" ".join([sketch_clip_split[0], sketch_clip_split[1], sketch_clip_split[3]]))
    return return_sketch

def parse_ori_program(sketch_clips):
    return_sketch = []
    for sketch_clip in sketch_clips:
        sketch_clip_split = sketch_clip.split(" ")
        if len(sketch_clip_split) == 1:
            return_sketch.append(sketch_clip.lower())
        elif len(sketch_clip_split) == 3:
            action, object = sketch_clip_split[:-1]
            action = action.lower()
            return_sketch.append(" ".join([action, object]))
        else:
            action, o1, o2 = sketch_clip_split[0], sketch_clip_split[1], sketch_clip_split[3]
            action = action.lower()
            return_sketch.append(" ".join([action, o1, o2]))
    return return_sketch


def parse_program(program):
    program_clips = program.split(", ")
    return_program = []
    for program_clip in program_clips:
        program_clip_split = program_clip.split(" ")
        if program_clip_split[1] == '<<none>>':
            return_program.append(program_clip_split[0])
        elif program_clip_split[3] == '<<none>>':
            return_program.append(" ".join([program_clip_split[0], program_clip_split[1]]))
        else:
            return_program.append(" ".join([program_clip_split[0], program_clip_split[1], program_clip_split[3]]))
    return return_program

def key_step_recall(sketch, prog):
    mapped_recoder = np.zeros(len(prog))
    match_num = 0
    for step in sketch:
        for r_i, ref_step in enumerate(prog):
            if mapped_recoder[r_i] == 1:
                continue
            if ref_step == step:
                mapped_recoder[r_i] = 1
                match_num += 1
                break
    return match_num/len(sketch)

def key_step_lcs(sketch, prog):
    lcs = LCS(sketch, prog)
    return len(lcs)/len(sketch)

def normal_lcs(sketch, prog, verbose=False):
    lcs = LCS(sketch, prog)
    # if verbose:
    #     print(lcs)
    return len(lcs)/max(len(sketch), len(prog))

def ori_lcs(sketch, prog):
    lcs = LCS(sketch, prog)
    return len(lcs)/len(prog)


def editDistanceWith2Ops(X, Y):
    # Find LCS
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n + 1)] for y in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                L[i][j] = 0
            elif (X[i - 1] == Y[j - 1]):
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    lcs = L[m][n]

    # Edit distance is delete operations +
    # insert operations.
    return (m - lcs) + (n - lcs)

def normal_edit(sketch, prog):
    editing_distance = editDistanceWith2Ops(sketch, prog)
    return editing_distance/max(len(sketch), len(prog))

def evaluate(info):
    programs = info['programs']
    ori_lcs_list = []

    for p_i, prog in enumerate(programs):
        path = prog['file_path']

        ori_path = path

        if "original" not in path:
            temp_path = path.replace("\\", "/").replace("//", "/")
            path_clips = temp_path.split("/")
            special_idx = path_clips.index("augment_programs")
            origin_path_clips = path_clips[:special_idx] + ["original_programs"] + ['executable_programs'] + path_clips[special_idx+3:-1]
            origin_path = "/".join(origin_path_clips) + '.txt'
        else:
            temp_path = path.replace("\\", "/").replace("//", "/")
            path_clips = temp_path.split("/")
            special_idx = path_clips.index("original_programs")
            origin_path_clips = path_clips[:special_idx+1] + ['executable_programs'] + path_clips[special_idx+2:]
            origin_path = "/".join(origin_path_clips)

        with open(ori_path, 'r') as f:
            ori_info = f.readlines()
        ori_gt_program = ori_info[4:]
        ori_gt_program = [step.strip() for step in ori_gt_program]
        parsed_ori_gt_program = parse_ori_program(ori_gt_program)

        with open(origin_path, 'r') as f:
            file_info = f.readlines()

        activity, dsec = file_info[0], file_info[1]
        original_program = file_info[4:]
        original_program = [step.strip() for step in original_program]
        parsed_ori_program = parse_ori_program(original_program)

        # double check, only remains the step in ori that exists in gt, exclude those steps that adapted to a specific scene
        new_parsed_ori_program = []
        for step in parsed_ori_program:
            if step in parsed_ori_gt_program:
                new_parsed_ori_program.append(step)

        sketch = prog['given_sketch']
        pred_prog = prog['pred_prog']
        gt_prog = prog['gt_prog']

        parsed_sketch = parse_sketch(sketch)
        parsed_prog = parse_program(pred_prog)
        parsed_gt_prog = parse_program(gt_prog)

        ori_lcs = normal_lcs(parsed_prog, new_parsed_ori_program)
        ori_lcs_list.append(ori_lcs)

        info['programs'][p_i]['rationality'] = ori_lcs

    # print("{} Rationality:{}".format(source_path, np.mean(ori_lcs_list)))
    info['rationlity'] = np.mean(ori_lcs_list)

    return info


if __name__ == "__main__":
    import glob
    source_path_list = ["/mnt/d/fuzzy/LangGuidedProg/src/output/Desc2ProgramGeo_r3_aHGN_dy8_trp18_lm1_r5_hflm4_lora3_v5_woac_oneE_dyn_v8_v22/testing_results-desc2program-best_inference.json"]

    for source_path in source_path_list:
        # source_path = "./Desc2ProgramGeo_r3_aHGN_dy7_trp18_lm1_r5_hflm4_lora3_GC_tianhuan_C_EXE.json"
        info = json.load(open(source_path, 'r'))
        info = evaluate(info)
        data = info
        data = evaluate(data)
        data['Metrics'] = {}
        data["Metrics"]['LCS'] = data["lcs"]
        data["Metrics"]['Rationlity'] = data["rationlity"]
        data["Metrics"]['Executability'] = data["executability"]
        data["Metrics"]['Completeness'] = data["total_f1"]
        print("|------Metrics------|\n|LCS          :{:.3f}|\n|Rationlity   :{:.3f}|\n|Executability:{:.3f}|\n|Completeness :{:.3f}|\n|-------------------|".format(data["lcs"], data["rationlity"], data["executability"], data["total_f1"]))

        json.dump(info, open("/mnt/d/fuzzy/temp.json", 'w'))