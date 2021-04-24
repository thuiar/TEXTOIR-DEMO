from django.shortcuts import render
from thedataset import models
from django.http import JsonResponse
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.csrf import csrf_exempt
# 分页
from django.core.paginator import Paginator
# 查询数据库结果转字典。
from django.forms.models import model_to_dict
from django.core import serializers
import pandas as pd
import os, sys, platform, shlex, subprocess, shutil, json


backend_engine_linux = '/../textoir/run_detect.py '
backend_engine_win = '\\..\\textoir\\run_detect.py '
base_runing_log_dir_linux = '/static/log/annotation/runing/'
base_runing_log_dir_win = '\\static\\log\\annotation\\runing\\'

# Create your views here.
@xframe_options_exempt
def data_annotation(request):#
    
    dataset_name = request.GET.get('dataset_name')
    model_detection = request.GET.get("model_detection")
    
    dataset_List = models.DataSet.objects.values()
    modelList_detection = models.Model_Tdes.objects.values().filter(type = 1)
    # example_list = models.Model_Test_Example.objects.values().filter(type="12")

    result = {}
    result['dataset_name'] = dataset_name
    result['dataset_List'] = dataset_List
    result['model_detection'] = model_detection
    result['modelList_detection'] = modelList_detection
    # result['exampleList'] = example_list

    return render(request,'annotation/data_annotation.html',result)


@csrf_exempt
def judgeStateDataAnnotation(request):
    dataset_id = request.POST['dataset_name']
    model_annotation = request.POST["model_detection"]
    errmsg = ['*', 'dataset_id:\t', dataset_id, '\nmodel_annotation:\t', model_annotation]
    print(' '.join(errmsg))
    
    base_runing_log_dir = '/static/log/annotation/runing/'
    obj = models.DataSet.objects.get(dataset_id=dataset_id)
    runing_log_dir = sys.path[0] + base_runing_log_dir + obj.dataset_name+'_'+model_annotation
    if os.path.exists(runing_log_dir):
        code = 201;
        msg = 'The combination is still running ！！！'
    # os.makedirs(runing_log_dir)
    # 取数据库查询是否已经存在
    exampleList = models.Data_Note_Annotation_Result.objects.values().filter(
        dataset_name=obj.dataset_name,method=model_annotation
    )
    # 数据库已有对应数据，询问是否再次运行,或直接从数据库显示结果
    if exampleList.count() != 0:
        code = 202;
        msg = 'The data already available ！！！'
    else:
        # 没有在运行中，也没有之前的结果，可以运行
        code = 200;
        msg = 'The combination is about to run ！！！'
    # msg = ' '.join(errmsg)
    return JsonResponse({'code':code,'msg':msg})


@csrf_exempt
def beginRunDataAnnotation(request):
    dataset_id = request.POST['dataset_name']
    model_annotation = request.POST["model_detection"]
    print('*', 'dataset_id:\t', dataset_id, '\nmodel_annotation:\t', model_annotation)
    # get dataset object
    obj = models.DataSet.objects.get(dataset_id=dataset_id)
    print(obj)
    # tun code params str
    param_list = [' --dataset', obj.dataset_name, '--method', model_annotation]
    para_str_python = ' '.join(param_list)
    # 拼接运行标注代码命令、log路径
    str_run = ''
    if platform.system() == 'Linux':
        runing_log_dir = sys.path[0] + base_runing_log_dir_linux + obj.dataset_name+'_'+model_annotation
        str_run = 'python ' + sys.path[0]+ backend_engine_linux +  para_str_python
    elif platform.system() == 'Windows':
        runing_log_dir = sys.path[0] + base_runing_log_dir_win + obj.dataset_name+'_'+model_annotation
        str_run = 'python ' + sys.path[0]+ backend_engine_win + para_str_python
    print('*'*20, '\n\runing_log_dir:', runing_log_dir, '\n\n')
    print('*'*20, '\n\nstr_run:', str_run, '\n\n')
    
    if os.path.exists(runing_log_dir):
        return JsonResponse({'code':201, 'msg':'The combination is still running ！！！'})
    os.makedirs(runing_log_dir)
    try:
        # run model
        str_run = shlex.split(str_run)
        process = subprocess.Popen(str_run)
        # #get pid
        # run_pid = process.pid
        

        ##get returncode ---wait runing state change
        process.communicate()
        run_type = process.returncode
        # run_type = 0  ## 测试用
        # ## save run_type to run log
        # 1--runing  0--finished others--failed
        if run_type == 0 or run_type == '0':
            # finished , save results to database
            save_annotation_results_2_database(obj=obj,model_annotation=model_annotation, 
                result_path=os.path.join(runing_log_dir,'results.csv'))
            print('run data annotation finished')
    except:
        return JsonResponse({'code': 400, 'msg': 'Run  Process Has An Error ！！'})
    finally:
        if os.path.exists(runing_log_dir):
            os.removedirs(runing_log_dir)

    return JsonResponse({'code':200, 'msg':''})

def save_annotation_results_2_database(result_path,obj,model_annotation):
    # 根据 file_path 读取文件，然后批量插入数据库表 Annotation_Result
    pd_data_list = pd.read_csv(result_path, sep= '\t' )
    data_list = pd_data_list.to_dict(orient='records')
    time_now = timezone.now().strftime("%Y-%m-%d %H:%M:%S")
    annotation_result_list = []
    for data_item in data_list:
        annotation_result_list.append(models.Annotation_Result(
            dataset_id=obj.dataset_id, dataset_name=obj.dataset_name, 
            sentences=data_item['sentences'], real_label=data_item['real_label'],
            predict_result=data_item['predict_result'], candidate_label_1=data_item['candidate_label_1'], 
            candidate_label_2=data_item['candidate_label_2'], candidate_label_3=data_item['candidate_label_3'], 
            create_time=time_now))
    models.Annotation_Result.objects.bulk_create(annotation_result_list)
    print('save results to database')

@csrf_exempt
def judgeStateDataExport2Disk(request):
    print('judgeStateDataExport2Disk')
    dataset_id = request.POST['dataset_name']
    model_annotation = request.POST["model_detection"]
    print('*', 'dataset_id:\t', dataset_id, '\nmodel_annotation:\t', model_annotation)
    base_runing_log_dir = '/static/log/annotation/runing/'
    obj = models.DataSet.objects.get(dataset_id=dataset_id)
    runing_log_dir = sys.path[0] + base_runing_log_dir + obj.dataset_name+'_'+model_annotation
    if os.path.exists(runing_log_dir):
        code = 201;
        msg = 'The combination is still running ！！！'
    # os.makedirs(runing_log_dir)
    # 取数据库查询是否已经存在
    exampleList = models.Data_Note_Annotation_Result.objects.values().filter(
        dataset_name=obj.dataset_name,method=model_annotation
    )
    # 数据库已有对应数据，从数据库保存结果
    if exampleList.count() != 0:
        # 保存结果
        export_results_2_disk(exampleList, obj.dataset_name, model_annotation)
        # 根据原 dataset 新增一条记录
        type = 2 #表示是自动标注的
        insert = models.DataSet.objects.create(dataset_name=obj.dataset_name+'_'+model_annotation, domain=obj.domain,
                                               class_num=obj.class_num, source=obj.source,
                                               local_path=obj.local_path, type=type,
                                               sample_total_num=obj.sample_total_num,
                                               sample_training_num=obj.sample_training_num,
                                               sample_validation_num=obj.sample_validation_num,
                                               sample_test_num=obj.sample_test_num,
                                               sentence_avg_length=obj.sentence_avg_length,
                                               sentence_max_length=obj.sentence_max_length,
                                               create_time=timezone.now().strftime("%Y-%m-%d %H:%M:%S"))
        insert.save()
        code = 200;
        msg = 'The data already available ！！！'
    else:
        # 没有在运行中，也没有之前的结果，可以运行
        code = 201;
        msg = 'The combination needs to be run ！！！'

    return JsonResponse({'code':code,'msg':msg})

def export_results_2_disk(dataList, dataset_name, method_name ):
    # 创建data下目录
    data_dir_path = os.path.join(sys.path[0],'../textoir/data')
    old_dataset_dir_path = os.path.join(data_dir_path, dataset_name)
    new_dataset_dir_path = os.path.join(data_dir_path, dataset_name, '_', method_name)
    os.makedirs(new_dataset_dir_path)
    # 复制train、vali
    for file_name in ['train.tsv', 'dev.tsv']:
        shutil.copy(os.path.join(old_dataset_dir_path, file_name), os.path.join(new_dataset_dir_path, file_name))
    # 从数据库保存test
    nameList = ['sentences', 'real_label']
    save_nameList = ['text', 'label']
    resultsListPd = pd.DataFrame( data=dataList)
    resultsListPd = pd[[nameList]].columns=save_nameList
    resultsListPd.to_csv(os.path.join(new_dataset_dir_path, 'test.tsv'), sep='\t')
    print('save results to disk')

@csrf_exempt
def getExampleList(request):
    # model_id = request.GET.get('model_id')
    dataset_id = request.GET.get('dataset_name')
    model_detection = request.GET.get("model_detection")
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    if dataset_id=='null':
        return JsonResponse({'code':0,'msg':'','count':0,'data':[]})
    # get Example list
    exampleList = models.Data_Note_Annotation_Result.objects.values().filter(dataset_id=dataset_id, method=model_detection).order_by('result_id')
    count = exampleList.count()
    # 分页
    paginator = Paginator(exampleList, limit)
    exampleList = paginator.get_page(page)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = list(exampleList)
    
    return JsonResponse(result)

@csrf_exempt
def updateResultByResultId(request):
    # model_id = request.GET.get('model_id')
    
    result_id = request.POST['result_id']
    val = request.POST['val']

    try:
        # ## save msg to Database
        record = models.Data_Note_Annotation_Result.objects.get(result_id=result_id)
        if record :
            record.real_label = val
            ## save msg to database
            record.save()
        else:
            return JsonResponse({'code': 400, 'msg': 'Edit Result Has An Error ！！'})
    except :
        # pass
        return JsonResponse({'code': 400, 'msg': 'Edit Result Has An Error ！！'})
    ## return msg
    return JsonResponse({'code':200,'msg':'Successfully Edit Result'})


@csrf_exempt
def getDatasetList(request):
    dataset_info_json_path = os.path.join(sys.path[0], '../frontend/static/jsons/data_annotation', 'dataset_info.json')
    with open(dataset_info_json_path, 'r') as load_f:
        dataset_info = json.load(load_f)
    if dataset_info.__contains__('dataset_list') == False :
        dataset_list = []
    else:
        dataset_list = dataset_info['dataset_list']
    # elif args.dataset not in dataset_info['dataset_list']:
    count = len(dataset_list)

    result = {}
    result['code'] = 0
    result['msg'] = ''
    result['count'] = count
    result['data'] = dataset_list
    
    return JsonResponse(result)

@csrf_exempt
def getClassListByDatasetNameAndClassType(request):
    dataset_name = request.GET.get('dataset_name')
    class_type = request.GET.get('class_type')
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    json_path = os.path.join(sys.path[0], 'static/jsons/data_annotation/', 'dataset_info.json')
    return_list = []
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        return_list = load_dict["class_list_"+dataset_name+"_"+class_type]
    
    count = len(return_list)
    paginator = Paginator(return_list, limit)
    return_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(return_list)
    return JsonResponse(results)

def getTextListByDatasetClassTypeLabelName(request):
    dataset_name = request.GET.get('dataset_name')
    class_type = request.GET.get('class_type')
    label_name = request.GET.get('label_name')
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    json_path = os.path.join(sys.path[0], 'static/jsons/data_annotation/', 'dataset_info.json')
    return_list = []
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        return_list = load_dict['text_list_'+str(dataset_name)+"_"+str(class_type)+"_"+str(label_name)]
    
    
    count = len(return_list)
    paginator = Paginator(return_list, limit)
    return_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(return_list)
    return JsonResponse(results)

def getTextListByDatasetForUnknown(request):
    dataset_name = request.GET.get('dataset_name')
    class_type = 'open' # all open === unknown
    page = request.GET.get('page')
    limit = request.GET.get("limit")
    json_path = os.path.join(sys.path[0], 'static/jsons/data_annotation/', 'dataset_info.json')
    load_dict = {}
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    # get class_list of all open by dataset_name
    class_list = load_dict["class_list_"+dataset_name+"_"+class_type]
    # Iterate over the class_list to get each text_list, and then join them to result_lsit
    result_list = []
    for class_item in class_list:
        label_name = class_item['label_name']
        # get text_list by dataset_name 、class_type、label_name
        text_list_item = load_dict['text_list_'+str(dataset_name)+"_"+str(class_type)+"_"+str(label_name)]
        result_list = result_list + text_list_item

    count = len(result_list)
    paginator = Paginator(result_list, limit)
    result_list = paginator.get_page(page)

    results = {}
    results['code'] = 0
    results['msg'] = ''
    results['count'] = count
    results['data'] = list(result_list)
    return JsonResponse(results)


@csrf_exempt
def judgeStateRunPipeline(request):
    dataset_id = request.POST['dataset_name']
    model_annotation = request.POST["model_detection"]
    errmsg = ['*', 'dataset_id:\t', dataset_id, '\nmodel_annotation:\t', model_annotation]
    print(' '.join(errmsg))
    
    base_runing_log_dir = '/static/log/annotation/runing/'
    obj = models.DataSet.objects.get(dataset_id=dataset_id)
    runing_log_dir = sys.path[0] + base_runing_log_dir + obj.dataset_name
    if os.path.exists(runing_log_dir):
        code = 201;
        msg = 'The combination is still running ！！！'
    # os.makedirs(runing_log_dir)
    # 取数据库查询是否已经存在
    exampleList = models.Data_Note_Annotation_Result.objects.values().filter(
        dataset_name=obj.dataset_name,method=model_annotation
    )
    # 数据库已有对应数据，询问是否再次运行,或直接从数据库显示结果
    if exampleList.count() != 0:
        code = 202;
        msg = 'The data already available ！！！'
    else:
        # 没有在运行中，也没有之前的结果，可以运行
        code = 200;
        msg = 'The combination is about to run ！！！'
    # msg = ' '.join(errmsg)
    return JsonResponse({'code':code,'msg':msg})
