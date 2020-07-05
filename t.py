
import re
 
 
title = '<a helf="www.baidu.com" title="河南省">你好</a>'
res = re.findall(r'<a.*?>(.*?)</a>', title)
print(res)



 
title = '[KG]趣秘[/KG][KG]乌白舞，[UNK]的,亲”克扬语她[S]部曲，作者，演剧恋[S]首歌手[S]所属系些罗齐体她性》是剧聊为作者[S]'
res = re.findall(r'\[KG\](.*?)\[\/KG\]', title)
print(res)


class Search:
    """
    匹配类
    """
    def __init__(self):
        pass

    def matching_pairs(self,mystr,tag):
        """
        匹配成对标签内的文本比如'<a helf="www.baidu.com" title="河南省">你好</a>' 或者div这种
        """
        res = re.findall(r'\['+tag+'.*?\](.*?)\[\/'+tag+'\]', mystr)
        return res

S=Search()
kg=S.matching_pairs(title,"KG")
print(kg)

        
