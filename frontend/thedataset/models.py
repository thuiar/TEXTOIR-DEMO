from django.db import models

# Create your models here.
# dataset_management 管理数据集信息
class DataSet(models.Model):
    dataset_id = models.AutoField(primary_key=True)
    dataset_name = models.CharField(max_length=255)
    domain = models.CharField(max_length=255)
    class_num = models.BigIntegerField()
    source = models.CharField(max_length=255)
    local_path = models.CharField(max_length=255)
    type = models.IntegerField()
    sample_total_num = models.BigIntegerField()
    sample_training_num = models.BigIntegerField()
    sample_validation_num = models.BigIntegerField()
    sample_test_num = models.BigIntegerField()
    sentence_max_length = models.IntegerField()
    sentence_avg_length = models.IntegerField()
    create_time = models.DateTimeField()

# data annotation 数据标注结果
class Annotation_Result(models.Model):
    result_id = models.AutoField(primary_key=True)
    dataset_id = models.IntegerField()
    dataset_name = models.CharField(max_length=255)
    sentences = models.CharField(max_length=255)
    real_label = models.CharField(max_length=255) ## 真实标签
    predict_result = models.CharField(max_length=255)
    candidate_label_1 = models.CharField(max_length=255, default='label_1')
    candidate_label_2 = models.CharField(max_length=255, default='label_2')
    candidate_label_3 = models.CharField(max_length=255, default='label_3')
    key_words = models.CharField(max_length=255)
    create_time = models.DateTimeField()

# model management 管理模型运行代码信息
class Model_Tdes(models.Model):
    model_id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=255)
    #model_source = models.CharField(max_length=255)
    paper_source = models.CharField(max_length=255)
    code_source = models.CharField(max_length=255)
    local_path = models.CharField(max_length=255)
    type = models.IntegerField()
    create_time = models.DateTimeField()
    
# 模型超参
class Hyper_parameters(models.Model):
    param_id = models.AutoField(primary_key=True)
    param_name = models.CharField(max_length=255)
    param_describe = models.CharField(max_length=255)
    default_value = models.CharField(max_length=255)
    value_type = models.CharField(max_length=255)
    run_value = models.CharField(max_length=255)
    model_id = models.IntegerField()

# 模型运行记录
class Run_Log(models.Model):
    log_id = models.AutoField(primary_key=True)
    dataset_name = models.CharField(max_length=255)
    model_name = models.CharField(max_length=255)
    # model_id = models.IntegerField()
    model_id = models.ForeignKey(Model_Tdes,related_name='model_tdes',db_constraint=False,on_delete=models.DO_NOTHING,blank=True)
    Annotated_ratio = models.CharField(max_length=255)
    Intent_Ratio = models.CharField(max_length=255)
    Local_Path = models.CharField(max_length=255)
    create_time = models.DateTimeField()
    type = models.IntegerField()
    run_pid = models.CharField(max_length=255)

    
# 模型超参
class run_log_hyper_parameters(models.Model):
    param_id = models.AutoField(primary_key=True)
    param_name = models.CharField(max_length=255)
    param_describe = models.CharField(max_length=255)
    default_value = models.CharField(max_length=255)
    value_type = models.CharField(max_length=255)
    run_value = models.CharField(max_length=255)
    log_id = models.IntegerField()


# 示例
class Model_Test_Example(models.Model):
    example_id = models.AutoField(primary_key=True)
    sentences = models.CharField(max_length=255)
    ground_truth = models.CharField(max_length=255) ## 真实标签
    predict_result = models.CharField(max_length=255)
    candidate_label_1 = models.CharField(max_length=255, default='label_1')
    candidate_label_2 = models.CharField(max_length=255, default='label_2')
    candidate_label_3 = models.CharField(max_length=255, default='label_3')
    key_words = models.CharField(max_length=255)
    type = models.IntegerField()
    
# 数据标注结果
class Data_Note_Annotation_Result(models.Model):
    result_id = models.AutoField(primary_key=True)
    dataset_id = models.IntegerField()
    dataset_name = models.CharField(max_length=255)
    sentences = models.CharField(max_length=255)
    real_label = models.CharField(max_length=255)
    predict_result = models.CharField(max_length=255)
    candidate_label_1 = models.CharField(max_length=255)
    candidate_label_2 = models.CharField(max_length=255)
    candidate_label_3 = models.CharField(max_length=255)
    key_words = models.CharField(max_length=255)
    create_time = models.DateTimeField()