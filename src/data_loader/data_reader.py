import json


class CDEEPipelineProcessor:
    def __init__(self, stage, max_length=None):
        self.stage = stage
        self.max_length = max_length

    def get_schema(self, schema_file):
        data = json.load(open(schema_file, "r"))
        if self.stage == "trigger":
            et2arg = data["et2arg"]
            ent2id = {et: idx for idx, et in enumerate(et2arg)}
            id2ent = {idx: et for idx, et in enumerate(et2arg)}
        if self.stage in ["argument", "argument_with_neg_new", "argument_embedding"]:
            roles = data["roles"]
            ent2id = {ent: idx for idx, ent in enumerate(roles)}
            id2ent = {idx: ent for idx, ent in enumerate(roles)}
        return ent2id, id2ent

    def read_data(self, input_file, data_type=None):

        def entity_search(entity, text, trigger_existed=set()):
            trig_start_inx = text.find(entity)
            if trig_start_inx != -1:
                if entity in trigger_existed:  # 如果entity已找到过一次
                    trig_start_inx_second = text.find(entity, trig_start_inx + 1)
                    if trig_start_inx_second != -1:
                        trig_start_inx = trig_start_inx_second  # 如果找到了第二个entity，就用第二个所在位置
                    else:
                        trig_start_inx = trig_start_inx_second  # 如果没找到第二个entity，认为不存在entity
                else:
                    trigger_existed.add(entity)
            return trig_start_inx, trigger_existed

        examples = []
        lines = json.load(open(input_file, "r"))
        # if data_type == 'train':
        #     lines = lines[:1587//10]
        for id_, line in enumerate(lines):
            if line.get("id"):
                id_ = line["id"]
            es = line.get("event", [])
            trigger_existed = set()

            if self.stage == "trigger":
                entities = []
                text = line["text"]
                if self.max_length:
                    text = text[:self.max_length]

                for e in es:
                    ent = e["core_name"]
                    ent_start_inx = text.find(ent)
                    ent_start_inx, trigger_existed = entity_search(ent, text, trigger_existed)
                    if ent_start_inx == -1:
                        continue
                    ent_type = e["tendency"] if e["tendency"] else "未知"
                    trig_new = {
                        "ent_text": ent,
                        "ent_start_inx": ent_start_inx,
                        "ent_type": ent_type,
                    }
                    if trig_new not in entities:
                        entities.append(trig_new)

                if entities:
                    examples.append({"id": id_, "text": text, "entity_list": entities})

            if self.stage == "argument":
                text_origin = line["text"]
                for e in es:
                    entities = []
                    ent = e["core_name"]
                    trig_start_inx, trigger_existed = entity_search(ent, text_origin, trigger_existed)
                    if trig_start_inx == -1:
                        continue
                    trig_end_inx = trig_start_inx + len(ent)
                    text = (
                        text_origin[:trig_start_inx]
                        + f"<T>"
                        + text_origin[trig_start_inx:trig_end_inx]
                        + f"</T>"
                        + text_origin[trig_end_inx:]
                    )

                    id_inline = f"{id_}_{trig_start_inx}"

                    for role in ["character", "anatomy_list"]:
                        arguments = e[role]
                        for argument in arguments:
                            argument_start_inx_right = text_origin.find(argument, trig_start_inx)
                            argument_start_inx_left = text_origin[:trig_start_inx].rfind(argument)
                            if (
                                argument_start_inx_right != -1
                                and argument_start_inx_left != -1
                            ):
                                right_distance = abs(trig_end_inx - argument_start_inx_right)
                                left_distance = abs(trig_end_inx - argument_start_inx_left)
                                if right_distance < left_distance:
                                    argument_start_inx = argument_start_inx_right
                                else:
                                    argument_start_inx = argument_start_inx_left
                            elif argument_start_inx_right != -1:
                                argument_start_inx = argument_start_inx_right
                            elif argument_start_inx_left != -1:
                                argument_start_inx = argument_start_inx_left
                            else:
                                continue
                            if argument_start_inx >= trig_end_inx:
                                argument_start_inx += 7
                            elif argument_start_inx >= trig_start_inx:
                                argument_start_inx += 3

                            argu_new = {
                                "ent_text": argument,
                                "ent_start_inx": argument_start_inx,
                                "ent_type": role,
                            }
                            if argu_new not in entities:
                                entities.append(argu_new)

                    examples.append(
                        {"id": id_inline, "text": text, "entity_list": entities}
                    )

            if self.stage == "argument_embedding":
                text_origin = line["text"]
                if self.max_length:
                    text_origin = text_origin[:self.max_length]
                for e in es:
                    entities = []
                    ent = e["core_name"]
                    trig_start_inx, trigger_existed = entity_search(ent, text_origin, trigger_existed)
                    if trig_start_inx == -1:
                        continue
                    trig_end_inx = trig_start_inx + len(ent)
                    id_inline = f"{id_}_{trig_start_inx}"

                    for role in ["character", "anatomy_list"]:
                        arguments = e[role]
                        for argument in arguments:
                            argument_start_inx_right = text_origin.find(argument, trig_start_inx)
                            argument_start_inx_left = text_origin[:trig_start_inx].rfind(argument)
                            if (
                                argument_start_inx_right != -1
                                and argument_start_inx_left != -1
                            ):
                                right_distance = abs(trig_end_inx - argument_start_inx_right)
                                left_distance = abs(trig_end_inx - argument_start_inx_left)
                                if right_distance < left_distance:
                                    argument_start_inx = argument_start_inx_right
                                else:
                                    argument_start_inx = argument_start_inx_left
                            elif argument_start_inx_right != -1:
                                argument_start_inx = argument_start_inx_right
                            elif argument_start_inx_left != -1:
                                argument_start_inx = argument_start_inx_left
                            else:
                                continue

                            argu_new = {
                                "ent_text": argument,
                                "ent_start_inx": argument_start_inx,
                                "ent_type": role,
                            }
                            if argu_new not in entities:
                                entities.append(argu_new)

                    examples.append(
                        {"id": id_inline, "text": text_origin, "entity_list": entities, 'sub_span': [trig_start_inx, trig_end_inx]}
                    )

        return examples


data_processors = {
    "cdee-pipeline": CDEEPipelineProcessor
}

task_types = {
    "cdee-pipeline": "ner"
}
