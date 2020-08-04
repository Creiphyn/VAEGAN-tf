# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:39:32 2020

@author: 72970
"""
import sys
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import matplotlib.pyplot as plt
from tensorflow.contrib.framework import arg_scope
from tensorflow.examples.tutorials.mnist import input_data
import re

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

class VAEGAN(object):
    def __init__(self, sess, input_dim=784, batch_size=64, output_dim=784, z_dim=2, conv_dim=5,fc_dim=1024):
        self.sess = sess    
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.conv_dim = conv_dim
        self.checkpointdir=""
        self.build_model()

    def build_model(self):

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        self.z_mean, self.z_log_sigma_sq = self.Encoder(self.x)        
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.exp(self.z_log_sigma_sq), eps))
        self.x_re = self.generator(self.z) #训练时批次生成x_re

        self.zp=tf.placeholder(tf.float32, [None, self.z_dim])
        self.x_p = self.generator(self.zp,reuse=True)   #训练时潜空间采样批次生成x_p

        self.Dx, self.Dx_logits = self.discriminator(self.x)        #带logits为未激活的结果，用于计算损失 Dx_logits：x的鉴别结果。                  
        self.Dx_p, self.Dx_p_logits = self.discriminator(self.x_p, reuse=True)    #Dx0_logits_tilde：x0重构的鉴别结果。
        self.Dx_re, self.Dx_re_logits = self.discriminator(self.x_re, reuse=True)    #Dxp_logits_tilde：随机采样生成的鉴别结果。
        

        self.xt=tf.placeholder(tf.float32, shape=[1, self.input_dim])  #测试时输入1个x潜变量
        self.zt_mean, self.zt_log_sigma_sq=self.Encoder(self.xt,reuse=True)        
        epst = tf.random_normal((1,self.z_dim), 0, 1, dtype=tf.float32)
        self.zt = tf.add(self.zt_mean, tf.multiply(tf.exp(self.zt_log_sigma_sq), epst))
        self.ret=self.sampler(self.zt,reuse=True)       #测试时生成1个重构

        self.sample=self.sampler(self.zp,reuse=True)   #测试时潜空间生成单个sample
        
        self.LL_loss = 0.5 * (  tf.reduce_sum(tf.square(self.Dx_logits - self.Dx_re_logits))) #/ (self.input_width*self.input_height)
        self.latent_loss =-0.5*tf.reduce_sum(1.0 + self.z_log_sigma_sq-tf.square(self.z_mean)-tf.exp(self.z_log_sigma_sq), axis=1)
        self.recon_loss = 0.5*tf.reduce_sum(tf.square(self.x - self.x_re)) #均方误差
        #总VAE损失
        self.vae_loss = tf.reduce_mean((32*self.latent_loss+self.recon_loss)/self.input_dim)

        #GAN损失：
        self.d_loss_real=0.5*( 
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_logits, labels=tf.ones_like(self.Dx)))
            )
        self.d_loss_fake=0.5*(
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_p_logits, labels=tf.zeros_like(self.Dx_p)))+
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_re_logits, labels=tf.zeros_like(self.Dx_re)))
            )
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = ( 0.5*( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_re_logits, labels=tf.ones_like(self.Dx_re)))
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx_p_logits, labels=tf.ones_like(self.Dx_p))))
                + tf.reduce_mean(self.recon_loss/self.input_dim))

        t_vars = tf.trainable_variables()

        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]    
        self.vae_vars = self.e_vars+self.g_vars

        self.lr_E = tf.Variable(0.25,trainable=False)
        self.lr_D = tf.Variable(0.15,trainable=False)
        self.lr_G = tf.Variable(0.25,trainable=False)
    
        self.vae_op = tf.train.MomentumOptimizer(self.lr_E,momentum=0.6).minimize(self.vae_loss,var_list=self.vae_vars)
        self.dis_op = tf.train.MomentumOptimizer(self.lr_D,momentum=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.gen_op = tf.train.MomentumOptimizer(self.lr_G,momentum=0.6).minimize(self.g_loss, var_list=self.g_vars)

        #self.vae_op = tf.train.AdamOptimizer(self.lr_E, beta1=0.5).minimize(self.vae_loss,var_list=self.vae_vars)
        #self.dis_op = tf.train.AdamOptimizer(self.lr_D, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        #self.gen_op = tf.train.AdamOptimizer(self.lr_G, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()  

    def Encoder(self,X,y=None,reuse=False):          
        with tf.variable_scope("Encoder") as scope:    
            if reuse:
                scope.reuse_variables()
                
            net = tf.reshape(X, [-1, 28, 28, 1])
            #net = tf.nn.elu(tf.layers.conv2d(net, filters=2,kernel_size=[5,5],strides=2,name='e_con1',padding='SAME',kernel_initializer= tf.contrib.layers.xavier_initializer()))
            #net = tf.nn.elu(tf.layers.conv2d(net, filters=4,kernel_size=[5,5],strides=2,name='e_con2',padding='SAME',kernel_initializer= tf.contrib.layers.xavier_initializer()))
            #net = tf.nn.elu(tf.layers.conv2d(net, filters=32,kernel_size=[5,5],strides=2,name='e_con3',padding='SAME',kernel_initializer= tf.contrib.layers.xavier_initializer()))#, padding='VALID') 输出64*784*16)
            net = tf.layers.flatten(net)#降维变成64*x1的张量
            #net=tf.nn.relu(tf.layers.dense(net, 128,name='e_h0'))
            #net= tf.nn.elu(tf.layers.dense(net,256,name='e_h0'))
            mean = tf.layers.dense(net, self.z_dim,name='e_4')   #输出为64*z_dim128均值张量
            log_sigma_sq = tf.layers.dense(net, self.z_dim ,name='e_5')   

            #eps = tf.random_normal([self.batch_size, self.z_dim], mean=0.0, stddev=1.0)
            #z_ = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
            return mean, log_sigma_sq

    def discriminator(self, h, y=None, reuse=False):    #鉴别器
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            #h=tf.reshape(X, [-1, 28, 28, 1])
            #h = tf.nn.relu(tf.layers.conv2d(h, filters=1,kernel_size=[3,3],strides=1,name='d_h1',padding='SAME',kernel_initializer= tf.contrib.layers.xavier_initializer()))  #二维卷积 输出64*88*16*32矩阵)
            #h = tf.nn.relu(tf.layers.conv2d(h, filters=8,kernel_size=[3,3],strides=1,name='d_h2',padding='SAME',kernel_initializer= tf.contrib.layers.xavier_initializer())) #二维卷积 输出64*88*16*32矩阵)
            #h = tf.nn.relu(tf.layers.conv2d(h, filters=32,kernel_size=[3,3],strides=1,name='d_h3',padding='SAME',kernel_initializer= tf.contrib.layers.xavier_initializer()))
           
            #h= tf.layers.dense(h,64,name='d_h0',activation=tf.nn.relu)
            h =tf.layers.dense(tf.reshape(h, [self.batch_size, -1]), 1, name='d_h4',activation=tf.nn.relu)
            #h =tf.layers.dense(X, 1, name='d_h5')   #先讲卷积压缩为64*（88*16*64）的一维向量，在通过linear输出64*1鉴别结果

            return tf.nn.sigmoid(h), h   # sigmoid激活将h4映射到0-1.第一项为激活后结果，第二项为未激活。

    def generator( self,z, y=None, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            #z= tf.layers.dense(z,256,name='g_h0',activation=tf.nn.relu) #输入一定维度z，输出大小为784*64 
            #h0= tf.reshape(z_, [-1, 28, 28,4])#把z_自动reshape成自动*88*16*64  

            #w1 = tf.get_variable('w1', [3,3,8,16],initializer=tf.random_normal_initializer(stddev=0.2))
           # h1 = tf.nn.conv2d_transpose(h0, w1,[self.batch_size, 28, 28,8],padding='SAME',strides=[1,1,1,1], name='g_h1')
            #h1 = tf.nn.relu(h1)  
            #w2 = tf.get_variable('w2', [3,3,4,8],initializer=tf.random_normal_initializer(stddev=0.2))
            #h2= tf.nn.conv2d_transpose(h0, w2,[self.batch_size, 28,28,4], padding='SAME',strides=[1,1,1,1], name='g_h2')
            #h2 = tf.nn.relu(h2)       #输入h1输出成None*88*16*16的h2
            #w3= tf.get_variable('w3', [3,3,1,4],initializer=tf.random_normal_initializer(stddev=0.2))
            #h4 = tf.nn.conv2d_transpose(h0, w3,[self.batch_size, 28,28,1], padding='SAME',strides=[1,1,1,1], name='g_h3')
            
            h4 = tf.layers.dense(tf.reshape(z, [self.batch_size,64]), 784, name='g_h4')
        return tf.reshape(tf.nn.tanh(h4),[self.batch_size,784])
    
    def sampler( self,z, y=None, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            #z= tf.layers.dense(z, 256,name='g_h0',activation=tf.nn.relu) #输入一定维度z，输出大小为784*64 
            
            #h0= tf.reshape(z_, [-1, 28, 28, 4])
            #w1 = tf.get_variable('w1', [3,3,8,16],initializer=tf.random_normal_initializer(stddev=0.2))
            #h1 = tf.nn.conv2d_transpose(h0, w1,[1, 28, 28, 8],padding='SAME',strides=[1,1,1,1], name='g_h1')
            #h1 = tf.nn.relu(h1)
            #w2 = tf.get_variable('w2', [3,3,4,8],initializer=tf.random_normal_initializer(stddev=0.2))
            #h2= tf.nn.conv2d_transpose(h0,  w2,[1, 28,28, 4], padding='SAME',strides=[1,1,1,1], name='g_h2')
            #h2 = tf.nn.relu(h2)       #输入h1输出成None*88*16*16的h2
            #w3= tf.get_variable('w3', [3,3,1,4],initializer=tf.random_normal_initializer(stddev=0.2))
            #h4 = tf.nn.conv2d_transpose(h0, w3,[1, 28,28, 1], padding='SAME',strides=[1,1,1,1], name='g_h3')
            h4 = tf.layers.dense(tf.reshape(z, [1,64]), 784, name='g_h4')

        return tf.reshape(tf.nn.tanh(h4),[1,784])



if __name__ == "__main__":

    MNIST_data =r'C:\Users\72970\Desktop\pretty-midi\MNIST_data'
    mnist = input_data.read_data_sets(MNIST_data, one_hot=True)
    with tf.Session() as sess:
        batch_size=100

        vaegan= VAEGAN(sess,input_dim=784, batch_size=batch_size, output_dim=784, z_dim=64, conv_dim=5,fc_dim=512)
        saver = tf.train.Saver()
        epochs=100
        al = np.zeros([epochs])
        bl= np.zeros([epochs])
        cl = np.zeros([epochs])
        aall=0
        ball=0
        call=0
        counter=0
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state("./")
        '''
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            saver.restore(sess,  os.path.join("./", ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
        '''
        # 开始训练;
        print("Start training...")
        # 批次数;
        total_batch =int(mnist.train.num_examples/batch_size)    #the number of train data:55000/64=
        # 开始迭代;
        
        for epoch in range(epochs):
            
            start_time = time.time()
            aall=0
            ball=0
            call=0
            for i in range(total_batch):
                batch_z = np.random.normal(0, 1, size=(batch_size , vaegan.z_dim))
                # 获取用于当前批次训练的数据;
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # 进行cost计算、训练操作;
                #for i in range(2):
                a,_=sess.run([vaegan.vae_loss,vaegan.vae_op], feed_dict={ vaegan.x: batch_xs})   #feed_dict替换值（喂入张量），方括号对应输出

                #if i%2==0:
                b,_=sess.run([vaegan.d_loss,vaegan.dis_op],feed_dict={vaegan.x:batch_xs,vaegan.zp:batch_z })
                #for i in range(2):
                c,_=sess.run([vaegan.g_loss,vaegan.gen_op],feed_dict={vaegan.x:batch_xs,vaegan.zp:batch_z })
                
                aall+=a
                ball+=b
                call+=c
                #print("\tvaeloss:{0},discrimitor_loss= {1},generator_loss= {2}".format(a,b,c))

            sample_z = np.random.normal(0, 1, size=(1,vaegan.z_dim))      
    
            x_reconstr = sess.run(vaegan.sample, feed_dict={vaegan.zp: sample_z})
            figg=plt.figure(figsize=(4,6))
            plt.subplot(1, 3, 1)
            plt.imshow(np.reshape(x_reconstr, [28, 28]), vmin=0, vmax=1, cmap="gray")
            plt.title("Random Sample")
            x_sample, _ = mnist.test.next_batch(1)
            plt.subplot(1, 3, 2)
            plt.imshow(np.reshape(x_sample[0],[28, 28]), vmin=0, vmax=1, cmap="gray")
            plt.title("Test input")
            x_reconstr = sess.run(vaegan.ret, feed_dict={vaegan.xt: np.reshape(x_sample[0],[1,784])})
            plt.subplot(1, 3, 3)
            plt.imshow(np.reshape(x_reconstr, [28, 28]), vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.savefig("./{0}_epochs_Recon.png".format(epoch+counter))
            
            bl[epoch]= ball/total_batch

            al[epoch]= aall/total_batch

            cl[epoch]= call/total_batch
            # 打印训练详情;
            print("\tEpoch:{0},time:{1} vae_loss= {2},discrimitor_loss= {3},generator_loss= {4},".\
                  format(epoch+counter,time.time() - start_time, al[epoch],bl[epoch],cl[epoch]))

        save_path = saver.save(sess, "./vaegan",global_step=epoch+counter)
        print("\tModel saved in file: {0}".format(save_path))
        
        nx=30
        ny=30
        zs=np.random.randn(900, 64)
        # 生成零矩阵;28*28是字体图片维度;
        canvas = np.zeros((28*30, 28*30))
        # 对应图像的零矩阵;
        xs_recon = np.zeros((900, 28*28))

        #按模型训练时的维度喂入数据，构成900个图片需要喂入9次
        for i in range(9):
            # 取zs第i到i+1单位批次的数据为z_mu,作为潜变量输入;
            z_mu = zs[100*i:100*(i+1), :]  #维度为100*2
            # 生成对应伪图像期望;
            x_mean = sess.run(vaegan.x_p, feed_dict={vaegan.zp: z_mu}) #输入隐变量z输出重建x
            # 设定xs_recon对应值(本来为零); 
            xs_recon[i*100:(i+1)*100] = x_mean
                
        n = 0
        # 开始 nx*ny 步的绘图(每次绘制一个28*28的图像);
        for i in range(nx):
            for j in range(ny):
                # 数据来自xs_recon;
                canvas[(ny-i-1)*28:(ny-i)*28, j*28:(j+1)*28] = xs_recon[n].reshape(28, 28)
                # 下一图;
                n = n + 1
                    
        # 图像大小设置为10*10;
        plt.figure(figsize=(10, 10))
        # 图的外观;
        plt.imshow(canvas,origin="upper", vmin=0, vmax=1, interpolation='none', cmap='gray')
        # 分割线;
        plt.tight_layout()
        # 存储结果;
        plt.savefig("./随机生成900张图片.png".format(epochs))


        
		

    fig=plt.figure(figsize=(4,3))
    plt.plot(al,label='vae loss')
    plt.plot(bl, label='discriminer loss')
    plt.plot(cl, label='generator loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig('loss.png')






