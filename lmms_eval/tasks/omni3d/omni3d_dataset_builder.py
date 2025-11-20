import json
import datasets
import os
import glob

class Omni3D(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image_identifier": datasets.Value("string"),
                "target_category": datasets.Value("string"),
                "full_json_str": datasets.Value("string"),
            })
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
             raise ValueError("data_files must be specified in dataset_kwargs")
             
        test_files = self.config.data_files.get("test")
        if not test_files:
             raise ValueError("No test files found in data_files for key 'test'")
             
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepaths": test_files},
            ),
        ]

    def _generate_examples(self, filepaths):
        files = []
        if isinstance(filepaths, str):
            files = glob.glob(filepaths)
        elif isinstance(filepaths, list):
            for path in filepaths:
                files.extend(glob.glob(path))
        
        idx = 0
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
                
                if not isinstance(data, list):
                    data = [data]
                    
                for item in data:
                    image_id = item.get("image_identifier", "")
                    object_grounding = item.get("object_grounding", [])
                    
                    # Get all unique categories present in the image
                    categories = set(obj["category"] for obj in object_grounding if "category" in obj)
                    
                    full_json_str = json.dumps(item)
                    
                    if not categories:
                        continue
                        
                    for cat in categories:
                        yield idx, {
                            "image_identifier": image_id,
                            "target_category": cat,
                            "full_json_str": full_json_str
                        }
                        idx += 1

