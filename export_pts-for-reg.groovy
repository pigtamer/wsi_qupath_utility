//This is actually two scripts, the first should be run in the image you are copying from
//and the second in the image you are copying to.
// 本文件用于将标注导出到文本文件或者从文本文件将标注导入到 Qupath 和 WSI 数据文件

// Execute this file in Qupath
// 注意本文件在 qupath 内执行


import static qupath.lib.gui.scripting.QPEx.*
import com.google.gson.GsonBuilder
import static qupath.lib.gui.scripting.QPEx.*

def ImgName = getProjectEntry().getImageName()
def cells = getAnnotationObjects().collect { new qupath.lib.objects.PathAnnotationObject(it.getROI(), it.getPathClass()) }
   
boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
File file = new File("/home/cunyuan/", 'pts-' + getProjectEntry().getImageName() + '.json')
file.withWriter('UTF-8') {
    gson.toJson(cells,it)
}