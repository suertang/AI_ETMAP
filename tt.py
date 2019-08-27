import tensorflow as tf
from tensorflow import keras
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def make_X(flow):
    '''生成预测的数据
    参数是nozzle flow
    以后会扩充到所有可输入参数
    '''
    tlf=pd.read_csv("Pred1.txt",index_col=0)
    tlf.drop(columns=["InjQ"],inplace=True)
    names=tlf.columns.values
    ETs=[e for e in range(80,1501,3)]
    RailP=[250000]+[p for p in range(300000,1700000,100000)]
    tf=pd.DataFrame()
    es=pd.Series(ETs)
    ps=pd.Series({"RailP":RailP})
    for p in RailP:
        ttf=pd.DataFrame(es,columns=["ET"])
        ttf['RailP']=p
        tf=tf.append(ttf,sort=False)
    tlf.drop(columns=["ET","RailP"],inplace=True)
    #print(names)
    tf=tlf.append(tf,sort=False)
    tf.fillna(method='ffill',inplace=True)
    tf.dropna(how='any',axis=0,inplace=True)
    tf = tf.reindex(columns=names)    
    tf['Nozzle Flow Rate_ cm3/30s@100bar']=flow
    
    return tf

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
# Add another:
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))
# Add another:
#model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(1))

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error
# Instantiates a toy dataset instance:


af=pd.read_csv("CRI.csv")
af.drop(columns=["Unnamed: 0","ID"],inplace=True)
X=af.iloc[:,:-1].values
y=af.InjQ.values
data,val_data, labels, val_labels=train_test_split(X,y,test_size = 0.05)


dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(1024)
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(128).repeat()

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  #keras.callbacks.TensorBoard(log_dir='./logs')
]
# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(data,labels, 
          epochs=20, 
          validation_data=(val_data,val_labels),
          #callbacks=callbacks
          )
plt.figure(figsize=(8, 21))

#output_pred = sess.run(output,{tf_x:x_pred})
flows=[350,400,500]
for i in range(len(flows)):    # 预测并画图
    ax = plt.subplot( len(flows),1,i+1)
    #plt.setp(ax)
    #plt.title("Nozzle:%d" % flows[i])
    X_new=make_X(flows[i])
    y_pred = model.predict(X_new) #sess.run(output,{tf_x:X_new,isTraing:False})
    #y_pred=model.predict(X_new.values)
    X_new['InjQ']=y_pred
    #X_new.loc[X_new.InjQ<0,'InjQ']=0
    pF=X_new.loc[:,["ET","RailP","InjQ"]]
    ff=pd.pivot_table(pF,index='ET',columns='RailP')
    
    ff.columns=ff.columns.droplevel()
    ff.plot(ax=ax,linewidth=1,
            title="Predicted Q-MAP for nozzle flow rate: {}".format(flows[i]))
plt.show()
