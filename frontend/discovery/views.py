# return data
from django.shortcuts import render
# download file
from django.http import FileResponse , JsonResponse
# return html
from django.views.decorators.clickjacking import xframe_options_exempt
# return Json , csv , shlex , subprocess , logging , os , sys , platform , shutil , stat , base64
import json , csv , shlex , subprocess , logging , os , sys , platform , shutil , stat , base64
from django.views.decorators.csrf import csrf_exempt
# time
from django.utils import timezone
# database tables
from thedataset import models
# page helper
from django.core.paginator import Paginator
# serializers
from django.core import serializers
# 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np
from io import BytesIO
from django.views import View
import matplotlib as mat
mat.use('Agg')
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.decomposition import PCA
# 分页
# 查询数据库
from django.db import connection
# 查询数据库结果转字典。
from django.forms.models import model_to_dict


@xframe_options_exempt
def model_management(request):
    return render(request,'discovery/model-list.html')


@csrf_exempt
def getModelList(request):
    model_name_select = request.GET.get('model_name_select')
    page = request.GET.get('page')
    limit = request.GET.get("limit")

    if model_name_select == None:
        model_name_select = ''
    
    modelList = models.Model_Tdes.objects.values().filter(model_name__contains=model_name_select,type=2).order_by('model_id')
    count = modelList.count()
    # 分页
    paginator = Paginator(modelList, limit)
    modelList = paginator.get_page(page)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(modelList)
    
    return JsonResponse(result)

@xframe_options_exempt
def model_management_details(request):
    model_id = request.GET.get('model_id')
    # print(model_id)
    obj = models.Model_Tdes.objects.get(model_id=model_id)
    # obj_param = models.Hyper_parameters.objects.get(model_id = model_id)
    paramList = models.Hyper_parameters.objects.filter(model_id=model_id)
    return render(request,'discovery/model-details.html',{'obj':obj,'paramList':paramList})


@xframe_options_exempt
def model_training(request):
    return render(request,'discovery/model-training-log-list.html')


@xframe_options_exempt
def getModelLogList(request):
    type_select = request.GET.get('type_select')
    dataset_select = request.GET.get("dataset_select")
    model_select = request.GET.get("model_select")
    page = request.GET.get('page')
    limit = request.GET.get("limit")

    
    logList = models.Run_Log.objects.values().filter(model_id__type=2)

    if dataset_select == None:
        dataset_select = ''
    if model_select == None:
        model_select = ''
    logList = logList.filter(dataset_name__contains=dataset_select,model_name__contains=model_select).order_by('-log_id')
    if type_select != '5':
        logList = logList.filter(type=type_select)

    # 分页
    paginator = Paginator(logList, limit)
    logList = paginator.get_page(page)
    count = paginator.count

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(logList)
    return JsonResponse(result)

@xframe_options_exempt
def toLogParameter(request):
    log_id = request.GET.get('log_id')
    paramList = models.run_log_hyper_parameters.objects.filter(log_id=log_id)
    return render(request,'discovery/model-training-log-parameter.html',{'model_id': log_id,'paramList':paramList})

@xframe_options_exempt
def toRunModel(request):
    
    # get model list
    modelList = models.Model_Tdes.objects.values().filter(type=2)
    datasetList = models.DataSet.objects.values()

    result = {}
    result['modelList'] = modelList
    result['datasetList'] = datasetList
    
    return render(request,'discovery/model-training-log-torun.html', result)


@csrf_exempt
def getParamListByModelId(request):
    model_id_select = request.GET.get('model_select')

    if model_id_select == None:
        return JsonResponse({'code':0,'msg':'','count':0,'data':[]})
    
    resultList = models.Hyper_parameters.objects.values().filter(model_id=model_id_select).order_by('param_id')
    
    paginator = Paginator(resultList, 100)
    resultList = paginator.get_page(1)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = paginator.count
    result['data'] = list(resultList)
    return JsonResponse(result)

@csrf_exempt
def add_model_training_log(request):
    #将模型的参数列表直接读取出来得到参数列表和参数的数量
    print('there is add_model_training_log')
    model_id = request.POST['model_id']
    dataset_name_select = request.POST['dataset_name_select']
    Known_Intent_Ratio = request.POST['Known_Intent_Ratio']
    Annotated_Ratio = request.POST['Annotated_Ratio']
    params = request.POST['params']
    paramsListJson = json.loads(params)

    modelItem = models.Model_Tdes.objects.get(model_id=model_id)
    model_name = modelItem.model_name
    
    # 拼接命令
    para_str_python = ' --dataset '+  dataset_name_select + ' --known_cls_ratio ' + Known_Intent_Ratio + ' --labeled_ratio ' + Annotated_Ratio + ' --method  '+ model_name
    for paramItem in paramsListJson:
        para_str_python = para_str_python + ' --' + paramItem['param_name'] + ' ' + paramItem['default_value']
        # print('param_id==',paramItem['param_id'],'\tparam_name==',paramItem['param_name'],'\tdefault_value==',paramItem['default_value'],'\trun_value==',paramItem['run_value'])
    print('para_str_python==',para_str_python)
    # 生成本地路径
    ## genernate local_path
    print('local_path===',sys.path[0])
    if platform.system() == 'Linux':
        local_path = sys.path[0]+'/static/log/discovery/add_model_training_log/running/'+  dataset_name_select + model_id + Annotated_Ratio + Known_Intent_Ratio +'/'
    elif platform.system() == 'Windows':
        local_path = sys.path[0]+'\\static\\log\\discovery\\add_model_training_log\\running\\'+  dataset_name_select + model_id + Annotated_Ratio + Known_Intent_Ratio +'\\'
    ## determine if the dataset exists
    if os.path.exists(local_path):
        return JsonResponse({'code':201,'msg':'The  Process Already Exists , Please Check It !!!'})

    try:
        run_logItem = models.Run_Log(                         
            dataset_name = dataset_name_select, 
            model_name = modelItem.model_name,
            model_id_id = model_id,
            Local_Path = local_path, 
            create_time = timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
            Annotated_ratio = Annotated_Ratio, 
            Intent_Ratio = Known_Intent_Ratio,
            type = 1 # runing state
            
            )
        run_logItem.save()
        # save msg to run_log_hyper_parameters
        for paramItem in paramsListJson:
            run_log_hyper_parametersItem = models.run_log_hyper_parameters(
                    param_name = paramItem['param_name'],param_describe = paramItem['param_describe'],
                    value_type = paramItem['value_type'],default_value = paramItem['default_value'],
                    run_value = paramItem['default_value'],log_id = run_logItem.log_id
                )
            run_log_hyper_parametersItem.save()

        os.makedirs(local_path)
        print("RUN"*10)
        str_run = ''
        if platform.system() == 'Linux':
            
            str_run = "python " + sys.path[0]+'/static/textoir/run_discover.py '+  para_str_python
        elif platform.system() == 'Windows':
            str_run = "python " + sys.path[0]+'\\static\\TEXTOIR\\textoir\\run_discover.py ' + para_str_python
        # run model
        print(str_run)
        print("RUN"*10)
        str_run = shlex.split(str_run)
        process = subprocess.Popen(str_run)
        
        # #get pid
        run_pid = process.pid
        # run_pid = 99999   ## 测试用
        ## save msg to run log
        updat = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(run_pid = run_pid)
        
        
        ##get returncode ---wait runing state change
        process.communicate()
        run_type = process.returncode
        # run_type = 0  ## 测试用
        # ## save run_type to run log
        # 1--runing  2--finished 3--failed
        if run_type == 0 or run_type == '0':
            type_after = 2
        else :
            type_after = 3
        # update runing state to run_log
        updat = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(run_pid = run_pid,type=type_after)

    except :
        if os.path.exists(local_path):
            os.removedirs(local_path)
            updat = models.Run_Log.objects.filter(log_id=run_logItem.log_id).update(type=3)
        return JsonResponse({'code': 400, 'msg': 'Run  Process Has An Error ！！'})

    if os.path.exists(local_path):
        os.removedirs(local_path)
    return JsonResponse({'code':200,'msg':'Successfully Running Process!'})



@xframe_options_exempt
def model_test(request):
    datasetList = models.DataSet.objects.values()
    modelList_discovery = models.Model_Tdes.objects.values().filter(type = 2)
    result = {}
    result['datasetList'] = datasetList
    result['modelList_discovery'] = modelList_discovery
    return render(request,'discovery/model-test.html' , result)

@xframe_options_exempt
def model_analysis(request):
    datasetList = models.DataSet.objects.values()
    modelList_discovery = models.Model_Tdes.objects.values().filter(type = 2)
    result = {}
    result['datasetList'] = datasetList
    result['modelList_discovery'] = modelList_discovery
    return render(request,'discovery/model-analysis.html' , result)

@csrf_exempt
def getModelAnalysisExampleData(request):
    def read(path):
        reader = csv.reader(open(path), delimiter = '\t', quotechar = None)
        lines = []
        for i,j in enumerate(reader):
            if i == 0 :
                continue
            lines.append(j[0]+'\n')
        return lines
    ##get base msg
    example_num = request.POST['example_num']
    str_path = ''
    if example_num == 'Example-1':
        str_path = 'test1_data'
    elif example_num == 'Example-2':
        str_path = 'test2_data'
    elif example_num == 'Example-3':
        str_path = 'test3_data'
    ## genernate local_path
    local_path = ''
    if platform.system() == 'Linux':
        local_path = sys.path[0]+'/static/data/test_data/'+  str_path + '/test.tsv'
    elif platform.system() == 'Windows':
        local_path = sys.path[0]+'\\static\\data\\test_data\\'+  str_path + '\\test.tsv'

    text = read(local_path)
    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Add Dataset','text':text})


@csrf_exempt
def modelAnalysisTest(request):

    print('there is discovery_test')
    model_discovery = request.POST['model_discovery']
    example_select_discovery = request.POST['example_select_discovery']

    print(model_discovery)
    print(example_select_discovery)
    
    #plt画图Kmeans
    model_name = "/home/lxt/tdes/pretrained_models/uncased_L-12_H-768_A-12"
    # 载入tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    #bert 分词   
    def bert_tokenizer(text):
        split_text = tokenizer.tokenize(text)
        return split_text

    
    tfidf_vectorizer = TfidfVectorizer(tokenizer=bert_tokenizer,lowercase=True)
  
    result_path ="/home/lxt/tdes/model/MSP/pred_result/result.txt"
    result = []
    with open(result_path,'r') as f:
        for line in f.readlines():
            data = line.split('\t\n')
            for strs in data:
                sub_str = strs.split('\t')
            if sub_str:
                result.append(sub_str[0]) 
    print(len(result))

    #需要进行聚类的文本集
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(result)
    # test_matrix = tokenizer.convert_tokens_to_ids(text_list)
    
    num_clusters = 3
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, 
                        init='k-means++',n_jobs=-1)
    #返回各自文本的所被分配到的类索引
    # result = km_cluster.fit_predict(tfidf_matrix)
    result_pred = km_cluster.fit_predict(tfidf_matrix)
    out = tfidf_matrix.toarray()

    #画图
    pca = PCA(n_components=2)
    tfidf_matrix = pca.fit_transform(out)

    #聚类边界
    #聚类中心降维
    center = pca.fit_transform(km_cluster.cluster_centers_)
    #循环画圆
    fig = plt.figure() 
    axes = fig.add_subplot(111)
    r = ["0.3","0.3","0.4"]
    color = ["#FF6666","#66CCCC","#333399"]
    theta = np.arange(0, 2*np.pi, 0.01)

    for i ,j in enumerate(r):  
        
        a = center[i][0]
        b = center[i][1]
        x = a + float(j)*np.cos(theta)
        y = b + float(j)*np.sin(theta)
        axes.plot(x, y, color = color[i])


    #画出聚类中心点
    plt.scatter(center[:,0],center[:,1], s=50,c='red', marker='+')
          
    #画出聚类散点
    plt.scatter(tfidf_matrix[:,0],tfidf_matrix[:,1], c = result_pred)
    plt.savefig("./static/img/discovery_img.png")
    plt.close()

    print('there is discovery_end')
    return JsonResponse({'code':200,'msg':'Successfully Discovery The Intent!'})














