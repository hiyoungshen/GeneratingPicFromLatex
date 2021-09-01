#!/usr/bin/python
# -*- coding:utf-8 -*-

from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
import re
from PyQt5.QtCore import Qt, QUrl, QTimer, QRect, QPoint, QSize, QObject, pyqtSlot
from PyQt5 import QtWidgets, QtWebChannel, QtCore
from PyQt5.QtGui import QPixmap
import sys
import time
from numpy.core.defchararray import partition
import pymysql
import json
import os
from bs4 import BeautifulSoup
import shutil
from collections import defaultdict
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from GeneratePaper import GeneratePaper
np.set_printoptions(threshold=np.inf)

sys.argv.append("--disable-web-security")

# Basic Config Area

# the max size of generated question is (max_w, max_h)
max_h = 100
max_w = 1080
# You can control the max char numbers of each formula generated
max_charnum = 80
# variables used for crop
global_strings = []
string_coords = []
formula_coords = []
global_formulas = []
mathWidth = 0
mathHeight = 0
saveIter = 10
# Dir indicates where the pictures save
# data_path indicates where the gt save
Dir = './data/img/'
GT = './data/gt/'
db_path = './data/db.txt'

# HTML Code
HTMLCode = """
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width">
    <title>MathJax example</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script>
        window.MathJax = {{
            tex: {{
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)'], ['\\\\[', '\\\\]']],
                        displayMath: []
                }},
            svg: {{
                    fontCache: 'global'
                }},
        }}
    </script>
    <!-- Todo: Change From MathJax v3 to MathJax v2 -->
    <!-- <script id="MathJax-script" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"> -->
    <!-- <script id="MathJax-script" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script> -->

    <script id="MathJax-script" src="https://cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script type="text/javascript" src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script type="text/x-mathjax-config">
        // http://docs.mathjax.org/en/v2.7-latest/options/output-processors/SVG.html
        MathJax.Hub.Config({{
            showProcessingMessages: false,
            messageStyle: "none",
            extensions: ["tex2jax.js"],
            jax: ["input/TeX", "output/HTML-CSS"],
            tex2jax: {{
                inlineMath:  [ ["$", "$"] ],
                displayMath: [ ["$$","$$"] ],
                skipTags: ['script', 'noscript', 'style', 'textarea', 'pre','code','a'],
                ignoreClass:"comment-content"
            }},
            "HTML-CSS": {{
                availableFonts: ["STIX","TeX", "Asana-Math", "Neo-Euler", "Gyre-Pagella", "Gyre-Termes", "Latin-Modern"],
                // availableFonts: ["Latin-Modern"],
                // availableFonts: ["Neo-Euler"],
                // availableFonts: ["Asana-Math"],
                showMathMenu: false
            }}
        }});
        MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    </script>
    <link rel="stylesheet" type="text/css" href="style.css" />

</head>

<body style="margin:0;top:0;">
    <div id="math" style="white-space: pre-line;margin:0;top:0;left:0;">{}</div>
</body>
<script type="text/javascript">
    let max_h = {}
    let max_w = {}
    // change the fontsize
    function ReShape() {{
            let element = document.getElementById("math")
            element.style.display = "inline"
            document.body.style.margin = '0'
            element.style.position = "absolute"

            let width = element.offsetWidth
            if (width > max_w) {{
                    element.style.fontSize = Math.floor(max_w / width * 100) - 5 + "%"
                }}
            let height = element.offsetHeight
            if (height > max_h) {{
                    element.style.fontSize = Math.floor(max_h / height * 100) - 5 + "%"
                }}
            element.style.display = ""
        }}
    // get text lines in HTML page
    function GetCoords() {{
            let Ele = document.getElementById("math")
            let nodes = Ele.childNodes

            function dealText(sNode) {{
                    let split_s = new Array()
                    let split_coords = new Array()
                    let s = sNode.nodeValue
                    if (s === '\\n' || s.replace(/\\s/g, "").length === 0) {{
                            return [split_s, split_coords]
                        }}
                    let start = 0
                    let range = document.createRange()
                    range.setStart(sNode, 0)
                    for (let i = 0; i !== sNode.length; i++) {{
                            range.setEnd(sNode, i + 1)
                            let rects = range.getClientRects()
                            // if range contains more than one text line
                            if (rects.length > 1) {{
                                    let rect = rects[0]
                                    split_s.push(s.substring(start, i))
                                    split_coords.push({{
                                        "x": rect.x,
                                        "y": rect.y,
                                        "width": rect.width,
                                        "height": rect.height
                                    }})
                                start = i
                                range.setStart(sNode, start)
                            }}
                    }}
                // add the last (or the only first) text line to result
                if (split_s.length === 0) {{
                        split_s.push(s)
                        let rects = range.getClientRects()
                        let rect = rects[0]
                        split_coords.push({{
                            "x": rect.x,
                            "y": rect.y,
                            "width": rect.width,
                            "height": rect.height
                        }})
                }} else if (start > 0 && start < sNode.length) {{
                    split_s.push(s.substring(start, sNode.length))
                    range.setStart(sNode, start)
                    range.setEnd(sNode, sNode.length)
                    let rects = range.getClientRects()
                    let rect = rects[0]
                    split_coords.push({{
                        "x": rect.x,
                        "y": rect.y,
                        "width": rect.width,
                        "height": rect.height
                    }})
            }}
        return [split_s, split_coords]
    }}
    let all_split_s = Array()
    let all_split_coords = Array()
    let formula_coords = Array()
    for (let i = 0; i != nodes.length; i++) {{
            let node = nodes[i]
            if (node.nodeName === "#text") {{
                    // split text region to textlines
                    let [split_s, split_coords] = dealText(node)
                    all_split_s = all_split_s.concat(split_s)
                    all_split_coords = all_split_coords.concat(split_coords)
                    // Todo: Change From MathJax v3 to MathJax v2
                }} else if (node.nodeName === "MJX-CONTAINER") {{
                    // record the formula region coords
                    let rect = node.firstElementChild.getClientRects()[0]
                    formula_coords.push({{
                        "x": rect.x,
                        "y": rect.y,
                        "width": rect.width,
                        "height": rect.height
                    }})
            }}
    }}
    window.backend.getCoords([all_split_s, all_split_coords, formula_coords])
}}
    function GetSize() {{
            // get the page size
            let element = document.getElementById("math")
            element.style.position = "absolute"
            element.style.display = "inline"
            element.style.fontSize = '150%'
            let width = element.offsetWidth
            let height = element.offsetHeight
            window.backend.getSize([width, height])
        }}
    function GetMath() {{
            // get all math formulas in page
            let element = document.getElementById("math")
            // Todo: Change From MathJax v3 to MathJax v2
            let formula_nodes = MathJax.startup.document.getMathItemsWithin(element);
            let formulas = Array()
            for (let i = 0; i != formula_nodes.length; i++) {{
                    formulas.push(formula_nodes[i].math)
                }}
            window.backend.getMath([formulas])
        }}
    window.onload = function () {{
            // use QWebChannel to transfer data from frontend to backend
            new QWebChannel(qt.webChannelTransport, function (channel) {{
                    window.backend = channel.objects.backend
                }})
        }}
</script>

</html>
"""

# If you want to get js offline, just use:
# <script id="MathJax-script" async
#   src="file:///MathJax-master/es5/tex-mml-chtml.js">
# </script>
# note that you should get zip from https://github.com/mathjax/MathJax/releases


# backend interface for QWebChannel
class HandleSize(QObject):
    @pyqtSlot(list)
    def getCoords(self, L):
        global global_strings
        global string_coords
        global formula_coords

        global_strings = L[0]
        string_coords = L[1]
        formula_coords = L[2]
        # print("Strings", global_strings)
        # print("String Coords", string_coords)
        # print("Formula Coords", formula_coords)

    @pyqtSlot(list)
    def getSize(self, L):
        global mathWidth
        global mathHeight

        mathWidth = L[0]
        mathHeight = L[1]

        # print("inside getSize: ", mathWidth, " ", mathHeight)

    @pyqtSlot(list)
    def getMath(self, L):
        global global_formulas
        global_formulas = L[0]
        # print("global_formulas: ", global_formulas)

class MainUi(QWebEngineView):
    def __init__(self):
        super().__init__()
        self.input_text = '1'
        self.output = 'webpage.png'
        
        # set background to transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(True)
        self.setStyleSheet("background:transparent")
        # self.page().settings().setAttribute(QWebEngineSettings.ShowScrollBars,
                                            # False)
        self.page().setBackgroundColor(Qt.transparent)
        # set callback function
        self.loadFinished.connect(self.on_loaded)
        self.resize(1200, 8000)

    def on_loaded(self):
        self.page().runJavaScript('GetSize()')
        self.page().runJavaScript('GetMath()')
        self.page().runJavaScript('GetCoords()')
        # get screenshot
        QTimer.singleShot(3000, self.saveimage)

    @staticmethod
    def gethtml(text):
        html = HTMLCode.format(text, max_h, max_w)
        # print(html)
        return html

    def convertQuestion(self, question, output):
        self.output = output
        html_code = self.gethtml(question)
        # print(html_code)
        # set the HTML code to Qt
        self.setHtml(html_code)

    # QImage转化为numpy
    def qtpixmap_to_cvimg(self, qtpixmap):
        qimg = qtpixmap.toImage()
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :4]
        # print(result.shape)
        # input()
        # print(result[:,:,0])
        # input()
        # print(result[:,:,1])
        # input()
        # print(result[:,:,2])
        # input()
        # print(result[:,:,3])
        # input()
        return result

    def saveimage(self):
        # grab specific region of page
        # region information was got from frontend
        fg=self.grab(QRect(QPoint(0, 0), QSize(mathWidth, mathHeight)))
        # fg.save(self.output, b'PNG')
        # print(fg.size())
        # input()
        fg_img=self.qtpixmap_to_cvimg(fg)
        b_channel, g_channel, r_channel,a_channel = cv2.split(fg_img)
        b_channel=255-b_channel
        g_channel=255-g_channel
        r_channel=255-r_channel
        a_channel=255-a_channel
        fg_img = cv2.merge((a_channel, a_channel, a_channel))

        bg_paper = GeneratePaper()
        max_edge=max(fg_img.shape[1], fg_img.shape[0])
        bg_img=bg_paper.texture(bg_paper.blank_image(width=max_edge, height=max_edge, background=230), sigma=4)
        bg_img=bg_paper.add_noise(bg_img, sigma=10)
        bg_img=bg_img[:fg_img.shape[0],:fg_img.shape[1],:]
        b_channel, g_channel, r_channel = cv2.split(fg_img)
        bg_img = cv2.merge((255-b_channel, 255-g_channel, 255-r_channel))
        # print(bg_img[:,:,0])
        # input()
        # print(bg_img[:,:,1])
        # input()
        # print(bg_img[:,:,2])
        # input()
        
        img=cv2.addWeighted(fg_img, 0.9, bg_img, 0.1, 0)
        # print(self.output)
        # input()
        cv2.imwrite(self.output, img)
        # print(fg_img)
        # print(fg_img.shape)
        self.app.quit()
        # cv2.imshow("image", fg_img)
        # cv2.waitKey(0)
        


def CalIOU(candidate_box, existing_box):
    # calculate the IoU of two boxes, IoU is used to
    # candidate_box : {'x': x_min, 'y': y_min, 'height': h, 'width': w}
    # exisiting_box : List of box
    def box2coord(box):
        # (xmin, ymin, xmax, ymax)
        coord = np.array([
            box['x'], box['y'], box['x'] + box['width'],
            box['y'] + box['height']
        ])
        return coord

    candidate = box2coord(candidate_box)
    exist = [box2coord(box) for box in existing_box]
    exist = np.stack(exist)

    xmin = np.maximum(candidate[0], exist[:, 0])
    ymin = np.maximum(candidate[1], exist[:, 1])
    xmax = np.minimum(candidate[2], exist[:, 2])
    ymax = np.minimum(candidate[3], exist[:, 3])

    interArea = np.maximum(0, xmax - xmin + 1) * np.maximum(0, ymax - ymin + 1)
    candidateArea = (candidate[2] - candidate[0] + 1) * (candidate[3] -
                                                         candidate[1] + 1)
    existArea = (exist[:, 2] - exist[:, 0] + 1) * (exist[:, 3] - exist[:, 1] +
                                                   1)

    iou = interArea / candidateArea
    return iou


# merge coords in the same line
def MergeCoords(formula_coords, string_coords, global_formulas,
                global_strings):
    Lines = defaultdict(list)

    def LineHeight(LineNum):
        # Line : List of Dict
        if LineNum == 0:
            return 0, 0
        Line = Lines[LineNum - 1]

        # the default textline height of MathJax 3 with charSize at 150% is 26.0
        max_height = 26.0 * LineNum
        mid_height = max_height - 13.0
        for coord, _, _ in Line:
            height = coord['y'] + coord['height']
            mid = coord['y'] + 0.5 * coord['height']
            if height > max_height:
                max_height = height
            if mid > mid_height:
                mid_height = mid
        # we give two height infromation for merge
        # max_height is the max height for all coords in this line
        # mid_height is used for merge
        max_height = max(max_height, 26.0 + LineHeight(LineNum - 1)[1])
        mid_height = max_height - 13.0
        return mid_height, max_height

    def mergeText(Line):
        sortedLine = sorted(Line, key=lambda x: x[0]['x'])
        text = ''.join([
            item[1] if item[2] == 'text' else '$' + item[1] + '$'
            for item in sortedLine
        ])
        return text

    # merge formula and text
    merge_coords_texts = [
        (formula_coord, formula, 'formula')
        for formula_coord, formula in zip(formula_coords, global_formulas)
    ]
    merge_coords_texts.extend([
        (string_coord, string, 'text')
        for string_coord, string in zip(string_coords, global_strings)
    ])

    # resort by the y coord
    if not len(merge_coords_texts) == 0:
        merge_coords_texts = sorted(merge_coords_texts,
                                    key=lambda x: x[0]['y'])
    # classify each coords to its line
    for coord, text, text_type in merge_coords_texts:
        y = coord['y']
        idx = 0
        while y > LineHeight(idx + 1)[0]:
            idx += 1
        Lines[idx].append((coord, text, text_type))

    resultCoords = []
    for idx, line in Lines.items():
        if line == []:
            continue
        min_x = min([coord['x'] for coord, _, _ in line])
        min_y = min([coord['y'] for coord, _, _ in line])
        max_x = max([coord['x'] + coord['width'] for coord, _, _ in line])
        max_y = LineHeight(idx + 1)[1]
        candidate = {
            'x': min_x,
            'y': min_y,
            'width': max_x,
            'height': max_y - min_y,
            'text': mergeText(line)
        }
        if len(resultCoords) > 0:
            iou = CalIOU(candidate, resultCoords)
            # if IoU > 0.5, we think candidate coord is redundant
            if np.any(iou > 0.5):
                continue
        if candidate['width'] == 0 or candidate['height'] == 0:
            continue
        resultCoords.append(candidate)
    return resultCoords


# delete tags like <div> <sup> etc.
def delete_html_tag(s):
    # # deal with <sup> </sup>
    # s = re.sub(
    #     r'(（[^\$<>]*)(?:\$*)([^<>：\$\s\u4e00-\u9fa5]+)(?:\$*)([^\$<>]*）)<sup[^>]*>([^<>]+)</sup>',
    #     r'$\g<1>\g<2>\g<3>^{\g<4>}$', s)
    # s = re.sub(
    #     r'(\([^\$<>]*)(?:\$*)([^<>：\$\s\u4e00-\u9fa5]+)(?:\$*)([^\$<>]*\))<sup[^>]*>([^<>]+)</sup>',
    #     r'$\g<1>\g<2>\g<3>^{\g<4>}$', s)
    # s = re.sub(r'([^\(\)（）<>：，{}\$\s\u4e00-\u9fa5]+)<sup[^>]*>([^<>]+)</sup>',
    #            r'$\g<1>^{\g<2>}$', s)
    # s = re.sub(r'<sup[^>]*>\s*</sup>', '', s)
    # # deal with <sub> </sub>
    # s = re.sub(
    #     r'(（[^\$<>]*)(?:\$*)([^<>：\$\s\u4e00-\u9fa5]+)(?:\$*)([^\$<>]*）)<sub[^>]*>([^<>]+)</sub>',
    #     r'$\g<1>\g<2>\g<3>_{\g<4>}$', s)
    # s = re.sub(
    #     r'(\([^\$<>]*)(?:\$*)([^<>：\$\s\u4e00-\u9fa5]+)(?:\$*)([^\$<>]*\))<sub[^>]*>([^<>]+)</sub>',
    #     r'$\g<1>\g<2>\g<3>_{\g<4>}$', s)
    # s = re.sub(r'([^\(\)（）<>：，{}\$\s\u4e00-\u9fa5]+)<sub[^>]*>([^<>]+)</sub>',
    #            r'$\g<1>_{\g<2>}$', s)
    # s = re.sub(r'<sub[^>]*>\s*</sub>', '', s)
    soup = BeautifulSoup(s, 'html.parser')
    s = ''.join(soup.findAll(text=True))
    return s


def main():
    # this function is used to generate transparent photos from the latex statements
    app = QtWidgets.QApplication(sys.argv)
    view = MainUi()
    view.app = app
    channel = QtWebChannel.QWebChannel(view)
    view.page().setWebChannel(channel)
    backend = HandleSize()
    channel.registerObject("backend", backend)
    view.show()
    # You should change this to your own setting
    # db = pymysql.connect('localhost', 'root', 'xxxxxx', 'latex')
    # cursor = db.cursor()
    # we get data from outputs tables
    # cursor.execute("SELECT question from outputs")
    # results = cursor.fetchall()
    with open("./sample.txt", "r") as f:
        results=f.readlines()
    # print(results[:10])

    # with open(db_path, 'w', encoding='utf-8') as f:
    #     f.write('\n'.join([
    #         # record[0] if record[0] is not None else '' for record in results
    #         record if record is not None else '' for record in results
    #     ]))

    # delete the old generated data
    if os.path.exists(GT):
        shutil.rmtree(GT)
    os.makedirs(GT)

    # Dir is used to save generated photos
    if os.path.exists(Dir):
        shutil.rmtree(Dir)
    os.makedirs(Dir)

    # start to render
    for index, question in enumerate(tqdm(results)):
        # for index, question in results:
        if os.path.exists(Dir + str(index) + '.png'):
            continue
        
        cur_question = ''
        try:
            # cur_question = question[0]
            cur_question = question
            if cur_question is None:
                continue

            # 清洗数据
            cur_question = cur_question.strip()
            cur_question = cur_question.replace('\u3000', ' ')
            cur_question = cur_question.replace('\\n', '\n')
            # replace \[ ... \] to $ ... $
            cur_question = re.sub(r'\\\[([^\]]+)\\\]', r'$\g<1>$', cur_question)
            cur_question = delete_html_tag(cur_question)
            cur_question = re.sub(r'["“”]', '', cur_question)
            cur_question = cur_question.replace("\\prime", '')
            # add spaces to '<' '>' '<=' '>='
            # change from '<' to ' < '
            # cause the html tags always use <>
            cur_question = re.sub(r"<(?!=)|<=|>(?!=)|>=", r" \g<0> ",
                                  cur_question)
            if (len(cur_question) == 0):
                # skip empty function after filter
                continue

            view.convertQuestion(cur_question, Dir + str(index) + '.png')
            sys.exit(app.exec_())

        except SystemExit:
            # except Exception as e:
            #     print(e)
            global global_strings, string_coords, formula_coords, global_formulas, mathWidth, mathHeight

            def ProcessText(s):
                # replace \\sum by \sum
                return repr(s.strip())[1:-1].replace('\\\\', '\\')

            # merge coords to textlines
            result_coords = MergeCoords(formula_coords, string_coords,
                                        global_formulas, global_strings)
            current_image_data = {
                "image_id": index,
                "image_path": str(index) + '.png'
            }
            current_image_data["original_text"] = ProcessText(cur_question)
            current_image_data["text"] = [
                ProcessText(text) for text in global_strings
            ]
            current_image_data["formula"] = [
                ProcessText(formula) for formula in global_formulas
            ]
            current_image_data["text_coords"] = string_coords
            current_image_data["formula_coords"] = formula_coords
            current_image_data["textline_coords"] = result_coords
            with open(GT + str(index) + '.json', 'w', encoding='utf-8') as f:
                json.dump(current_image_data, f, ensure_ascii=False)

            # reset Global Variables
            global_strings = []
            string_coords = []
            formula_coords = []
            global_formulas = []
            mathWidth = 0
            mathHeight = 0
            continue


if __name__ == '__main__':
    main()
