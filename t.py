
import re
 
from tkitMatch import Match


title = '[KG]克扬语她[S]部曲，作者，演剧恋[S]首歌手[S]所属系些罗齐体她性》是剧聊为作者[/KG][KG]乌白舞，[UNK]的,亲”克扬语她[S]部曲，作者，演剧恋[S]首歌手[S]所属系些罗齐体她性》是剧聊为作者[S]'
S=Match()
kg=S.matching_pairs(title,"KG")
print(kg)

for it in kg:
    print(it.split("[S]"))
# ['趣秘']