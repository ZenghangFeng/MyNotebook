import pymysql


class MyDataBase:
    def __init__(self, host, database, user, password):
        # 数据库基本连接信息 =============================================
        self.host = host
        self.database = database
        self.user = user
        self.password = password

        # 以下变量用于数据库初始化 =========================================
        self.str_create_db = "CREATE DATABASE IF NOT EXISTS " + self.database
        self.str_ddl_table_1 = """
                                CREATE TABLE IF NOT EXISTS steel_qa_sts (
                                    batch_code VARCHAR(50) PRIMARY KEY COMMENT '批次号',
                                    steel VARCHAR(50) COMMENT '钢材名称',
                                    batch_num INT COMMENT '批次数量',
                                    batch_qualified_num INT COMMENT '批次合格数量',
                                    batch_qualified_rate DECIMAL(38,6) COMMENT '批次合格率',
                                    pd_date VARCHAR(10) COMMENT '生产日期',
                                    pd_line VARCHAR(50) COMMENT '生产线'
                                ) COMMENT='钢材质量情况统计表' CHARACTER SET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                                """
        self.str_ddl_table_2 = """
                                CREATE TABLE IF NOT EXISTS pd_line_info (    
                                    pd_line VARCHAR(50) COMMENT '生产线',
                                    device_name VARCHAR(50) COMMENT '设备名称',
                                    device_no VARCHAR(50) COMMENT '设备编号',
                                    admin_name VARCHAR(50) COMMENT '设备管理员'
                                ) COMMENT='产品线信息表' CHARACTER SET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                                """
        self.str_truncate_table_1 = "TRUNCATE TABLE steel_qa_sts"
        self.str_truncate_table_2 = "TRUNCATE TABLE pd_line_info"
        self.str_insert_table_1 = """
                                INSERT INTO steel_qa_sts (batch_code,steel,batch_num,batch_qualified_num,batch_qualified_rate,pd_date,pd_line) VALUES
                                     ('100001','热轧钢',100,99,0.990000,'2025-03-23','热轧'),
                                     ('100002','锻钢',30,30,1.000000,'2025-04-25','锻造'),
                                     ('100003','锻钢',60,55,0.916667,'2025-03-30','锻造'),
                                     ('100004','热轧钢',120,110,0.916667,'2025-04-26','热轧'),
                                     ('100005','锻钢',300,296,0.986667,'2025-04-27','锻造'),
                                     ('100006','热轧钢',200,198,0.990000,'2025-04-27','热轧');
                                """
        self.str_insert_table_2 = """
                                INSERT INTO pd_line_info (pd_line,device_name,device_no,admin_name) VALUES
                                     ('热轧','六辊轧机-01','zj_06_01','李工'),
                                     ('热轧','六辊轧机-02','zj_06_02','李工'),
                                     ('热轧','十二辊轧机-01','zj_12_01','张工'),
                                     ('锻造','三十吨锻造机-01','dzj_30_01','张工'),
                                     ('锻造','三十吨锻造机-02','dzj_30_02','张工'),
                                     ('锻造','六十吨锻造机-01','dzj_60_01','刘工');
                                """

    def db_connect(self):
        conn = pymysql.connect(host=self.host, user=self.user, password=self.password)
        cursor = conn.cursor()
        return conn, cursor

    def db_close_connect(self, conn, cursor):
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    def sql_query(self, sql):
        # 建立数据库连接 --------------------------------------
        conn, cursor = self.db_connect()
        # 执行查询语句 ----------------------------------------
        if "CREATE DATABASE" != sql[0:15]:
            cursor.execute("USE " + self.database)
        cursor.execute(sql)
        # 提交事务 -------------------------------------------
        conn.commit()
        # 关闭连接 -------------------------------------------
        self.db_close_connect(conn=conn, cursor=cursor)

    def db_init(self):
        str_list = [self.str_create_db, self.str_ddl_table_1, self.str_ddl_table_2, self.str_truncate_table_1, self.str_truncate_table_2, self.str_insert_table_1, self.str_insert_table_2]
        for s in str_list:
            self.sql_query(s)
        print("---------------数据库初始化完成-----------------")


# if __name__ == "__main__":
#     host = "localhost"
#     database = "test1"
#     user = "root"
#     password = "123456"
#
#     my_db = MyDataBase(host=host, database=database, user=user, password=password)
#     my_db.db_init()
