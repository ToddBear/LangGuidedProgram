vl2nl_mapper = {
    "Find": "Find {}",
    "Walk" : "Walk to {}",
    "Run": "Run to {}",
    "TurnLeft": "Turn left",
    "TurnRight": "Turn right",
    "Sit": "Sit on {}",
    "StandUp": "Stand up",
    "Grab": "Grab {}",
    "Open": "Open {}",
    "Close": "Close {}",
    "Put": "Put {} on {}",
    "PutIn": "Put {} in {}",
    "PutBack": "Put {} back to {}",
    "PutOn": "Put on {}",
    "SwitchOn": "Switch on {}",
    "SwitchOff": "Switch off {}",
    "Drink": "Drink {}",
    "TurnTo": "Turn to {}",
    "LookAt": "Look at {}",
    "PointAt": "Point at {}",
    "Wipe": "Wipe {}",
    "PutOff": "Put off {}", # 放下物体
    "Greet": "Greet {}",
    "Drop": "Drop {}",
    "Read": "Read {}",
    "Touch": "Touch {}",
    "Lie": "Lie on {}",
    "Pour": "Pour {} into {}",
    "Type": "Type on {}",
    "Watch": "Watch {}",
    "Move": "Move {}",
    "Wash": "Wash {}",
    "Squeeze": "Squeeze {}",
    "PlugIn": "Plug in {}",
    "PlugOut": "Plug out {}",
    "Cut": "Cut {}",
    "Eat": "Eat {}",
    "Sleep": "Sleep",
    "WakeUp": "Wake up",
    "PutObjBack": "Put back {}",
    "Push": "Push {}", # [Push] <button> (1)
    "Scrub": "Scrub {}", # [Scrub] <teeth> (1)
    "Rinse": "Rinse {}", # [Rinse] <rag> (1)
    "Pull": "Pull {}", # [Pull] <laptop> (1)
    "Release": "Release {}", 
}

vl2nl_mapper = {key.lower():vl2nl_mapper[key] for key in vl2nl_mapper.keys()}

def transfrom_into_sentence(program):
    atomic_action_in_nl_list = []
    for atomic_action in program:
        atomic_split = atomic_action.split(" ")
        action = atomic_split[0].replace("]", "").replace('[', "")
        param2 = None
        param_num = 0
        if len(atomic_split) >= 3:
            param1 = atomic_split[1].replace("<", "").replace(">", "")
            param_num = 1
            if len(atomic_split) >3 :
                param_num = 2
                param2 = atomic_split[3].replace("<", "").replace(">", "")
        # atomic_action_nl_pattern = vl2nl_mapper[action]
        atomic_action_nl_pattern = vl2nl_mapper_complete["{}_{}".format(action, param_num)]
        if len(atomic_split) >= 3:
            atomic_action_in_nl = atomic_action_nl_pattern.format(param1) if param2 is None else atomic_action_nl_pattern.format(param1, param2)
        else:
            atomic_action_in_nl = atomic_action_nl_pattern
        atomic_action_in_nl_list.append(atomic_action_in_nl)

    return atomic_action_in_nl_list


def transfrom_format_into_sentence(program):
    atomic_action_in_nl_list = []
    for atomic_action in program:
        atomic_split = atomic_action.split(" ")

        action = atomic_split[0].replace("]", "").replace('[', "").replace("<", "").replace(">", "")
        param1 = atomic_split[1].replace("<", "").replace(">", "")
        param2 = atomic_split[2].replace("<", "").replace(">", "")

        if action == 'sos' or action == 'eos' or action == 'none':
            atomic_action_in_nl_list.append("")
            continue

        param_num = 0
        if param1 != 'sos' and param1 != 'eos':
            param_num += 1
        else:
            param1 = ""
        if param2 != 'sos' and param2 != 'eos':
            param_num += 1
        else:
            param2 = ""

        # atomic_action_nl_pattern = vl2nl_mapper[action]
        atomic_action_nl_pattern = vl2nl_mapper["{}".format(action)]
        under_fill_parameter_num = atomic_action_nl_pattern.count("{}")

        if under_fill_parameter_num == 1:
            atomic_action_in_nl = atomic_action_nl_pattern.format(param1)
        elif under_fill_parameter_num == 2:
            atomic_action_in_nl = atomic_action_nl_pattern.format(param1, param2)
        else:
            atomic_action_in_nl = atomic_action_nl_pattern

        atomic_action_in_nl_list.append(atomic_action_in_nl)

    return atomic_action_in_nl_list