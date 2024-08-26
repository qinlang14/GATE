from Data_Processor import DataProcessor

from OpendialKG.Preprocess.csv2json import convert
from OpendialKG.Preprocess.render import get_render
from OpendialKG.Preprocess.knowledge_base import get_knowledge_base
from OpendialKG.Preprocess.dataset import OpendialkgDataset

from WoW.Preprocess.knowledge_base import kb
from WoW.Preprocess.dataset import WoWDataset

processor = DataProcessor()

# OpendialKG
# 1
data_file = "OpendialKG/OriginalData/opendialkg.csv"
out_file = "OpendialKG/Preprocess/Intermediate/opendialkg.json"
convert(data_file, out_file)

# 2
get_render()

# 3
get_knowledge_base()

# 4
opendialkg = OpendialkgDataset()
opendialkg.get_data(origin=True, k_hop=2)
opendialkg.get_data(origin=False, k_hop=2)

# 5
processor.embedding(data_name="OpendialKG")
processor.add_label(data_name="OpendialKG")

# WoW
# 1
kb()

# 2
wow = WoWDataset()
wow.get_data()

# 3
processor.embedding(data_name="WoW")
processor.add_label(data_name="WoW")