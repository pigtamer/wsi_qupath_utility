//https://forum.image.sc/t/exporting-polygons-of-cell-detections-objects-along-in-csv/44959

import static qupath.lib.gui.scripting.QPEx.*
import com.google.gson.GsonBuilder
import static qupath.lib.gui.scripting.QPEx.*

def ImgName = getProjectEntry().getImageName()
def cells = getCellObjects()

boolean prettyPrint = true
def gson = GsonTools.getInstance(prettyPrint)
File file = new File('/Users/cunyuan/'+ImgName+'.json')
file.withWriter('UTF-8') {
    gson.toJson(cells,it)
}