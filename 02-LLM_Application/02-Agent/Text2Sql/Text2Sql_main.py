from openai import OpenAI
from Text2Sql_database import MyDataBase
from Text2Sql_agent import MyVanna


############################################################
# 1. 进行数据库初始化
############################################################
# 数据库连接参数 ----------------------------------------------
host = "localhost"
database = "test1"
user = "root"
password = "123456"
# 进行数据库初始化 --------------------------------------------
my_db = MyDataBase(host=host, database=database, user=user, password=password)
my_db.db_init()


############################################################
# 2. 创建Text2Sql Agent 实例
############################################################
# 创建大模型对象 ==============================================
api_key="your api key"
base_url="https://api.deepseek.com"
client = OpenAI(api_key=api_key, base_url=base_url)

# 创建Vanna对象 ==============================================
config={"model": "deepseek-reasoner"}
vn = MyVanna(client=client, config=config)

# 连接数据库 =================================================
vn.connect_to_mysql(host=host, dbname=database, user=user, password=password, port=3306)

# 训练 ======================================================
vn.train(ddl=my_db.str_ddl_table_1)
vn.train(ddl=my_db.str_ddl_table_2)

# 对话 ======================================================
question = "锻钢的100002批次和100003批次的合格率分别是多少？"
# sql, res, fig = vn.ask(question=question)
# 生成数据库查询语句 -------------------------------------------
sql_generated = vn.generate_sql(question=question)
# 执行sql进行查询，返回结果 -------------------------------------
res_df = vn.run_sql(sql_generated)
# 生成绘图的python程序，是基于plotly模块的绘图 --------------------
plot_code = vn.generate_plotly_code(question=question, sql=sql_generated, df_metadata=str(res_df.columns))
# 执行绘图的程序，显示图表 ---------------------------------------
fig = vn.get_plotly_figure(plotly_code=plot_code,df=res_df)
fig.show()
""""""