# 知识图谱KG  ——微信公众号【亦一君兮】

## “The world is not made of strings , but is made of things.”  ——Amit Singhal

## 图谱概念

### 图谱定义

- 知识图谱旨在描述真实世界中存在的各种实体或概念。其中，每个实体或概念用一个全局唯一确定的ID来标识，称为它们的标识符。每个属性-值对用来刻画实体的内在特性，而关系用来连接两个实体，刻画它们之间的关联。--王昊奋老师
- 实体、属性、关系

### 图谱分类

- 按领域

	- 通用知识图谱

		- 大众版，一般不涉及特别深的行业知识和专业内容，解决科普类、常识类等问题
		- 业务场景：面向全领域，用于面向互联网的搜索、推荐、问答等，强调广度

	- 特定领域知识图谱：行业、垂直知识图谱

		- 专业版，一般涉及某个行业或细分领域深入研究的专业内容
		- 面向特定领域，准确率要求高，强调深度

- 按知识类型

	- 概念图谱、百科图谱、常识图谱、词汇图谱、学术图谱、人物图谱等

### 知识工程发展历程

- 前知识工程时期——>专家系统——>万维网1.0——>群体智能——>知识图谱

## 图谱技术节点

### 知识表示与建模

- 知识表示

  知识表示是将现实世界中存在的知识转换成计算机可识别和处理的内容，是一种描述知识的数据结构，用于对知识的一种描述或约定。

	- 1. 基于符号

	  可分为早期知识表示方法，如一阶谓词逻辑表示法、产生式规则表示法、框架表示法，和语义网知识表示方法，如RDF、RDFS、OWL。

	- 2. 基于表示学习

		- a. 前期表示学习模型

		  距离模型：如结构表示（structured embedding, SE）
		  单层神经网络模型（single layer model, SLM）
		  语义匹配能量模型（semantic matching energy, SME）
		  双线性模型：如隐变量模型（latent factor model, LFM），DISTMULT模型（LFM的简化形式）
		  张量神经网络模型（neural tensor network, NTN）
		  矩阵分解模型：如RESACL模型

		- b. 翻译模型：Trans系列

		  TransE、TransH、TransR、TransD、TranSparse等。

- 知识建模

  知识建模是指建立知识图谱的数据模型，即采用什么样的方式来表达知识，构建一个本体模型对知识进行描述。在本体模型中构建本体的概念、属性以及概念之间的关系。具体地，将业务问题按照知识图谱约定的一些模式进行业务抽象以及业务建模，主要是概念本体、领域实体、关系定义、事件定义等。
  本体的概念，本体是用于描述事物的本质的，维基百科对本体的定义是这样的，即对于特定领域真实存在的实体的类型、属性，以及它们之间的相互关系的一种定义。

	- 1. 自顶向下

	  构建知识图谱时首先定义数据模式即本体，一般通过领域专家人工编制。从最顶层的概念开始定义，然后逐步细化，形成结构良好的分类层次结构。

	- 2. 自底向上

	  首先对现有实体进行归纳组织，形成底层的概念，再逐步向上抽象形成上层的概念。多用于开放域知识图谱的构建。

### 知识获取

知识图谱中的知识来源于结构化、半结构化和非结构化的信息资源。知识获取即通过获取这些不同来源、不同结构的知识，形成结构化的知识并存储到知识图谱中。当前的知识抽取主要针对文本数据进行，需要解决的抽取问题包括：实体抽取、关系抽取、属性抽取和事件抽取。

资料（模型过于繁多）：
https://www.cnblogs.com/sandwichnlp/p/12020066.html
https://www.cnblogs.com/sandwichnlp/p/12049829.html
http://shomy.top/2018/02/28/relation-extraction/
https://github.com/roomylee/awesome-relation-extraction
https://zhuanlan.zhihu.com/p/91762831
https://zhuanlan.zhihu.com/p/142615620
http://www.shuang0420.com/2018/09/15/知识抽取-实体及关系抽取/
http://www.shuang0420.com/2018/10/15/知识抽取-事件抽取/
https://github.com/smilelight/lightKG
https://github.com/loujie0822/DeepIE

- 实体识别

  命名实体识别，简称NER，是自然语言处理中的一项基础任务，应用广泛。命名实体一般指的是文本中具有特定意义或者指代性强的实体，通常包括人名、地名、组织机构名、专有名词等，NER则是从非结构化的输入文本中抽取出上述实体，并且可以按照业务需求识别出更多类别的实体。

	- 1. 基于规则

		- a. 自定义词典识别

		  自定义词典
		  构建实体词典进行匹配，通常可采用Trie树/Hashtable、Double array trie、AC自动机，另外可借助分词和句法工具保证词典匹配出的结果与文本分词后的结果兼容。
		  
		  优势：速度快、精准识别，适合实体明确、数量级小的场景
		  劣势：词典构建和更新麻烦，且词典有限，无法发现新词

		- b. 规则模板

		  人工编写规则，一般由具有一定领域知识的专家手工构建。
		  
		  优势：小数据集上可以达到较高的准确率和召回率。
		  劣势：随着数据集增大，规则集的构建周期变长，后期维护成本较高，而且模式的可移植性差，缺乏语义泛化能力，召回率普遍较低。

	- 2. 基于统计模型

	  资料https://zhuanlan.zhihu.com/p/71190655

		- a. Hidden Markov Models (HMM)

		  https://github.com/lipengfei-558/hmm_ner_organization
		  未知参数：实体类型
		  优势：考虑了上下文信息
		  劣势：假设了可观测变量之间相互独立，限制观测变量是词语本身，限制了特征的选择，如词频、位置等。

		- b. Maximum Entropy Markov Models (MEMM)

		  http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf
		  生成模型变为判别模型，只计算给定可观测变量下隐藏变量的概率。
		  优势：打破了HMM的观测独立假设
		  劣势：由于局部归一化，会造成标注偏差问题

		- c. Conditional Random Fields

		  http://www.ai.mit.edu/courses/6.891-nlp/ASSIGNMENT1/t11.1.pdf
		  https://github.com/bojone/crf/
		  将最大熵马尔可夫模型里面的条件概率转化为特征函数的形式，分解为两部分，转移特征和状态特征。
		  优势：打破了观测独立假设，克服了标注偏差
		  劣势：训练速度较慢

	- 3. 基于深度学习

		- a. 序列标注

		  各模型的性能比较（数据来自论文）
		    
		  
		  从目前已发表的论文来看，Bert_Flat和后面提到的Bert_MRC是NER任务中效果最佳的，Flat尚未开源，后面有机会将会对目前主流的和最新的NER算法/模型做一下实验对比和总结。
		   详见：https://zhuanlan.zhihu.com/p/142615620
		  
		  补充|Flat已开源（正在看）
		  https://github.com/LeeSureman/Flat-Lattice-Transformer

			- 字符级别

				- (Bi)LSTM + CRF

				  https://arxiv.org/pdf/1603.01360.pdf
				    
				  
				  NER任务中比较主流的模型，包含三层结构，Embedding层（主要有词向量、字向量和一些额外特征）、双向LSTM层和最后的CRF层，如图所示。

				- Bert + (Bi)LSTM + CRF

				  https://github.com/EOA-AILab/NER-Chinese
				  使用Bert模型替换了原网络中的word2vec部分，从而构成Embedding层，同样使用双向LSTM层以及CRF层做序列预测。

			- 融合词汇信息

			  由于中文分词存在误差，基于字符的模型效果通常好于基于词汇（经过分词）的方法。但基于字符的NER没有利用词汇信息，而词汇边界对于实体边界通常起着至关重要的作用，所以说引入词汇信息，对中文NER任务很有必要。
			  
			  基于词汇增强的中文NER主要分为两条主线：
			  a. Dynamic Architecture：设计一个动态框架，能够兼容词汇输入
			  b. Adaptive Embedding：基于词汇信息，构建自适应Embedding
			  
			  ACL2020中的两篇论文FLAT和Simple-Lexicon分别对应于Dynamic Architecture和Adaptive Embedding，这两种方法相比于其他方法：
			  1）能够充分利用词汇信息，避免信息损失；
			  2）FLAT不去设计或改变原生编码结构，设计巧妙的位置向量就融合了词汇信息；Simple-Lexicon对于词汇信息的引入更加简单有效，采取静态加权的方法可以提前离线计算。

				- Dynamic Architecture

				  Dynamic Architecture范式通常需要设计相应结构以融入词汇信息。

					- Lattice LSTM

					  Lattice LSTM  2018
					  https://arxiv.org/pdf/1805.02023.pdf
					  基于词汇增强方法的中文NER的开篇之作，提出了一种Lattice结构以融合词汇信息。
					   
					  
					  Lattice是一个有向无环图，词汇的开始和结束字符决定了其位置。Lattice LSTM结构则是融合了词汇信息到原生的LSTM中，如图。
					    
					  
					  Lattice LSTM引入了一个word cell结构，对于当前的字符，融合以该字符结束的所有word信息。如‘桥’融合了‘大桥’和‘长江大桥’。
					  
					  优势：引入词汇信息，提升了NER性能
					  劣势：计算性能低下，不能batch并行化，Batch size只能为1；信息损失，每个字符只能获取以它为结尾的词汇信息，对于其之前的词汇信息没有持续记忆。如对于‘大’，其无法获得‘inside’的‘长江大桥’信息。另外由于RNN特性，采取Bi LSTM时其前向和后向的词汇信息不能共享；可迁移性差，只适配于LSTM，不具备向其他网络迁移的特性。

					- LR-CNN

					  LR -CNN：CNN-Based Chinese NER with Lexicon Rethinking  2019
					  https://pdfs.semanticscholar.org/1698/d96c6fffee9ec969e07a58bab62cb4836614.pdf
					  
					  针对Lattice LSTM采取RNN结构，导致其不能充分利用GPU进行并行化，以及RNN仅仅依靠前一步的信息输入，而非全局信息，无法有效处理词汇信息冲突的问题，LR -CNN提出：
					  
					  a. Lexicon-Based CNNs：采取CNN对字符特征进行编码，感受野大小为2提取bi-gram特征，堆叠多层获得multi-gram特征，同时采用注意力机制融入词汇信息
					  
					  b. Refining Networks with Lexicon Rethinking：对于词汇信息冲突的问题，LR-CNN采取rethinking机制增加feedback layer来调整词汇信息的权值，即将高层特征作为输入通过注意力模块调节每一层词汇特征分布。如图，高层特征得到的‘广州市’和‘长隆’会降低‘市长’在输出特征中的权重分布。最终对每一个字符位置提取对应的调整词汇信息分布后的multi-gram特征，输入CRF层解码。
					    
					  
					  优势：相比Lattice LSTM加速三倍多
					  劣势：计算复杂，不具备可迁移性

					- CGN

					  CGN:Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network  2019
					  https://www.aclweb.org/anthology/D19-1396.pdf
					  
					  针对Lattice LSTM存在信息损失，尤其是无法获得‘inside’的词汇信息问题，提出了CGN --基于协作的图网络。不再详述。

					- LGN

					  LGN:A Lexicon-Based Graph Neural Network for Chinese NER  2019
					  https://www.aclweb.org/anthology/D19-1096.pdf
					  
					  针对Lattice LSTM使用RNN结构仅仅依靠前一步的信息输入，而不是全局信息，从而无法解决词汇冲突的问题，提出LGN。如下图所示，将每一个字符作为节点，匹配到的词汇信息构成边，通过图结构实现局部信息的聚合（Multi- Head Attention），并增加全局节点进行全局信息融入。

					- FLAT

					  FLAT: Chinese NER Using Flat-Lattice Transformer  2020  https://arxiv.org/pdf/2004.11795.pdf
					  
					  众所周知，Transformer采取全连接的自注意力机制可以很好地捕捉长距离依赖，由于自注意力机制对位置是无偏的，因此Transformer引入位置向量来保持位置信息。受到位置向量表征的启发，FLAT设计了一种巧妙的position encoding来融合Lattice 结构，如上图，对于每一个字符和词汇都构建两个head position encoding和tail position encoding，重构了原有的Lattice结构。因此，FLAT可以直接建模字符与所有匹配的词汇信息间的交互，例如，‘药’可以匹配到词汇‘人和药店’和‘药店’。
					    
					  
					  因此，我们可以将Lattice结构展平，将其从一个有向无环图展平为一个平面的Flat-Lattice Transformer结构，由多个span构成：每个字符的head和tail是相同的，每个词汇的head和tail是跳跃的。Flat结构如下图所示。

				- Adaptive Embedding

				  Adaptive Embedding范式仅在Embedding层对于词汇信息进行自适应，后面通常接入LSTM+CRF或其他通用网络，其与模型无关，具备可迁移性。

					- WC-LSTM

					  WC -LSTM: An Encoding Strategy Based Word-Character LSTM for Chinese NER Lattice LSTM  2019
					  https://pdfs.semanticscholar.org/43d7/4cd04fb22bbe61d650861766528e369e08cc.pdf?_ga=2.158312058.1142019791.1590478401-1756505226.1584453795
					    
					  
					  针对Lattice LSTM中每个字符只能获取以它为结尾的词汇数量是动态的、不固定的，从而导致不能Batch并行化的问题，WC -LSTM采取词汇编码策略，将每个字符为结尾的词汇信息进行固定编码表示，即每一个字符引入的词汇表征是静态的、固定的，如果没有对应的词汇则用<PAD>代替，从而实现Batch并行化。
					  
					  优势：相对于Lattice LSTM，可以并行化
					  劣势：信息损失，无法获得‘inside’的词汇信息，建模能力有限，效率低

					- Multi-digraph

					  Multi-digraph: A Neural Multi-digraph Model for Chinese NER with Gazetteers 2019 
					  https://www.aclweb.org/anthology/P19-1141.pdf
					  
					  不同于其他论文引入词汇信息是基于一个词表（由Lattice LSTM提供，没有实体标签，通常需要提取对应的word embedding），本文引入词汇信息的方式是利用实体词典（Gazetteers，含实体类型标签）。图结构，不再详述。

					- Simple-Lexicon

					  Simple-Lexicon：Simplify the Usage of Lexicon in Chinese NER  2020
					  https://arxiv.org/pdf/1908.05969.pdf
					  
					  本文提出一种在Embedding层简单利用词汇的方法，对比了三种不同的融合词汇的方式。
					  a. Softword
					  b. Extend Softword
					  c. Soft-lexicon
					    
					  
					  最佳表现是Soft-lexicon，它对当前字符，依次获取BMES对应所有词汇集合，然后再进行编码表示。

			- 融合字形信息

				- Glyce

				  文章：Glyph-vectors for Chinese Character Representations
				  https://arxiv.org/abs/1901.10125
				  https://github.com/ShannonAI/glyce  香侬官方开源
				  
				  Glyce，一种汉字字形的表征向量，将汉字当作图像，使用CNN去获取语义表征：
				  a. 使用历时的汉字字形（如金文、隶书、篆书等）和不同的书写风格（如行书、草书等）来丰富字符图像的象形信息，从而更全面地捕捉汉字的语义特征。
				  b. 提出Tianzige-CNN（田字格）架构，专门为中文象形字符建模而设计。
				  c. 添加图像分类的辅助损失，通过多任务学习增强模型的泛化能力。
				    
				  为了将字形信息和传统的词向量和字向量结合起来，香侬针对词级别的任务和字级别的任务，提出了两种Glyce模型结构。模型的结构总览见上图。在字级别的任务中，可以首先通过上述方法得到一个Glyph Embedding，然后和对应的Char Embedding结合起来，具体可以使用concat，highway或者全连接，然后送入下游任务。在词级别的任务中，首先有一个Word Embedding，对这个词而言，分别对其中的每一个字进行字级别的embedding操作，把所有的字级别的embedding用max-pooling结合起来，然后再和这个Word Embedding结合，送入下游任务。
				    
				    
				  
				  上面两图分别是Glyce与Bert的结合，以及将Glyce-Bert应用于不同任务时的情况。个人尚未验证效果，但训练很耗时，不适合快速落地。

		- b. 指针网络

		  指针网络通常应用于神经机器阅读理解，转化为2个n元分类预测头指针和尾指针，但这种方法只能抽取单一片段；对于实体识别，同一类型实体可能会存在多个，应转化为n个2元分类预测头指针和尾指针。

			- 堆叠式指针网络

			  简介：构造多类别的堆叠式指针网络，每一层指针网络对应一种实体类别。（了解后再补充...）

			- MRC QA + 单层指针网络

			  简介：构建query问题（即转化为机器阅读理解QA问题），问题即指代所要抽取的实体类型，同时引入了先验语义知识，从而能够在小数据集下、迁移学习下表现更好。
			    
			  
			  文章：A Unified MRC Framework for Named Entity Recognition
			  https://arxiv.org/pdf/1910.11476.pdf
			  https://github.com/qiufengyuyi/sequence_tagging 大神复现
			  https://github.com/ShannonAI/mrc-for-flat-nested-ner 香侬官方开源
			  	
			  基于MRC的命名实体识别任务，MRC分为三部分，问题、答案和文本，首先需要将序列标注数据变成这种适用于阅读理解任务的三元组，主要是对于要抽取的每一类实体，构造关于该类实体的Query，然后将需抽取实体的原始文本作为content，预测该类实体在content中的位置，一般是start_position和end_position，即实体的开始位置和结束位置。
			  
			  如下所示，{ 问题：“找出人名和虚构的人物形象”，答案：“彭真”，文本：“这次会议上彭真同志当选为全国人大常委会副委员长兼秘书长”}。
			    
			  
			  相比传统的序列标注做法，MRC方式做NER的好处在于引入了Query这个先验知识，比如对于PER这个类别，我们构造一个Query：找出人名和虚构的人物形象。模型通过attention机制，对于Query中的人名，人物形象学习到了person的关注信息，然后反哺到content中的实体信息捕捉中。
			    
			  模型以Bert为基础，输入时将问题和文本concat在一起，输出开始和结束位置，从而得到实体位置。
			  
			  Bert+MRC的优势：（官方说法）
			  1. 在训练数据较少的情况下优于Bert-Tagger；
			  2. 能够在统一框架下处理flat NER和嵌套NER任务；
			  3. 具有更好的zero-shot学习能力，可以预测训练集中不存在的标签，可应用在迁移学习上；
			  4. Query编码了有关要提取的实体类别的先验信息，有潜在能力消除相似类的歧义。

				- Bert + MRC

		- c. 片段排列 + 分类

		  序列标注和Span抽取的方法都是停留在token-level进行NER，间接去提取span-level的特征。而基于片段排列的方式，显示的提取所有可能的片段排列，由于选择的每一个片段都是独立的，因此可以直接提取span-level的特征去解决重叠实体问题。
		  
		  对于含T个token的文本，理论上共有 T(T+1)/2 种片段排列。如果文本过长，会产生大量的负样本，在实际中需要限制span长度并合理削减负样本。

- 关系抽取

  利用多种技术自动从文本中发现命名实体之间的语义关系，将文本中的关系映射到实体关系三元组上。研究难点是关系表达的隐含性（关系不一定明显）、关系的复杂性（二元或多元）、语言的多样性（关系有多种表述形式）。

	- 1. 基于模板匹配

	  模板匹配是关系分类中比较常见的方法，使用一个模板库对输入文本中的两个给定实体进行上下文匹配，如果满足模板对应关系，则作为实体对之间的关系。
	  
	  优势：实现与构建简单，适用于小规模特定领域，效果好
	  劣势：召回率低、可移植性差，不适合大规模及通用领域

		- a. 人工模板

		  主要用于判断实体间是否存在上下位关系。上下位关系的自然语言表达方式相对有限，采用人工模板就可以很好地完成关系分类。

		- b. 统计模板

		  无需人工构建，主要基于搜索引擎进行统计模板抽取。具体地，将已知实体对作为查询语句，抓取搜索引擎返回的前n个结果文档并保留包含该实体对的句子集合，寻找包含实体对的最长字串作为统计模板，保留置信度较高的模板用于关系分类。

	- 2. 基于深度学习有监督方法

		- a. 流水线关系抽取Pipeline

		  Pipeline方法中，关系抽取通常转化为一个分类问题，前提是关系类型数目是确定的，如果关系无法完全列举，则可将关系抽取转化为一个相似度匹配任务（问句与属性/关系的匹配）。
		  
		  案例|NER --> RE (Bert + Dense) 关系分类
		  优势：把关系抽取当作一个简单的分类问题，高质量监督数据下的模型的准确率会很高。
		  劣势：需要大量的人力和时间成本做数据标注，难以拓展新的关系类别，模型较为脆弱，泛化能力有限。
		  https://github.com/yuanxiaosc/Entity-Relation-Extraction
		  
		  a-1 Matching the Blanks: Distributional Similarity for Relation Learning
		    
		  本文基于Bert，采用6种不同结构进行实体对的pooling，然后将pooling进行关系分类或关系相似度计算，结论是(f)效果最好。
		  优势：提出了关系抽取预训练任务Matching the blanks，在少样本关系抽取任务上效果提升明显，适用于少数据的场景。
		  https://arxiv.org/pdf/1906.03158.pdf
		  https://github.com/zhpmatrix/BERTem
		  
		  a-2 Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers
		  https://arxiv.org/pdf/1902.01030.pdf
		  
		  a-3 Simultaneously Self-Attending to All Mentions for Full-Abstract Biological Relation Extraction
		  https://www.aclweb.org/anthology/N18-1080.pdf
		  
		  分析：将实体抽取和关系抽取作为串联任务，先抽取实体，再抽取关系，建模相对更简单，易于实现。但当作两个独立的任务会存在一些问题。
		  一，误差积累：关系抽取任务的结果严重依赖于实体抽取的结果；
		  二，关系重叠：对于一对多问题，串联模型无法提供较好的解决方案；
		  三，交互缺失：忽略了两个子任务之间的内在联系和依赖关系。

		- b. 联合抽取Joint Model

		  联合模型，即将两个子模型统一建模，从而进一步利用两个任务之间的潜在信息，以缓解错误积累的问题。联合抽取的难点在于如何加强实体模型和关系模型之间的交互，比如实体模型和关系模型的输出之间存在着一定的约束，在建模的时候考虑到此类约束将有助于联合模型的性能。

			- 共享参数

			  通过共享参数（共享输入特征或者内部隐层状态）实现联合，这种方法对于子模型没有限制，但由于使用独立的解码方法，导致实体模型和关系模型之间的交互不强。
			    
			  b-1 依存结构树: End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures
			  https://www.aclweb.org/anthology/P16-1105.pdf
			  https://github.com/mkacan/entity-relation-extraction
			  
			  b-2 指针网络: Going out on a limb: Joint Extraction of Entity Mentions and Relations without Dependency Trees
			  https://pdfs.semanticscholar.org/bbbd/45338fbd85b0bacf23918bb77107f4cfb69e.pdf?_ga=2.119149259.311990779.1584453795-1756505226.1584453795
			  https://github.com/Luka0612/JEAR
			  
			  b-3_1 Copy机制+seq2seq: Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism
			  https://www.aclweb.org/anthology/P18-1047.pdf
			  https://github.com/xiangrongzeng/copy_re
			  b-3_2 CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning
			  https://arxiv.org/abs/1911.10438
			  https://github.com/WindChimeRan/CopyMTL
			  
			  b-4 多头选择机制+sigmoid: Joint entity recognition and relation extraction as a multi-head selection problem
			  https://arxiv.org/pdf/1804.07847.pdf
			  https://github.com/bekou/multihead_joint_entity_relation_extraction
			  
			  b-5 SPO问题+指针网络: Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy
			  https://yubowen-ph.github.io/files/2020_ECAI_ETL/ETL.pdf
			  https://github.com/yubowen-ph/JointER
			  
			  b-6 多轮对话+强化学习: Entity-Relation Extraction as Multi-Turn Question Answering
			  https://arxiv.org/pdf/1905.05529.pdf
			  https://zhuanlan.zhihu.com/p/65870466
			  
			  b-7 输入端的片段排列: Span-Level Model for Relation Extraction
			  https://www.aclweb.org/anthology/P19-1525.pdf
			  
			  b-8 输出端的片段排列: SpERT：Span-based Joint Entity and Relation Extraction with Transformer Pre-training
			  https://arxiv.org/pdf/1909.07755.pdf

			- 联合解码

			  复杂的联合解码方法，可以加强实体模型和关系模型的交互，如整数线性规划等。
			  
			  b-9 Joint extraction of entities and relations based on a novel tagging scheme
			  https://arxiv.org/pdf/1706.05075.pdf
			  https://github.com/zsctju/triplets-extraction
			  https://github.com/gswycf/Joint-Extraction-of-Entities-and-Relations-Based-on-a-Novel-Tagging-Scheme
			  
			  b-10 Joint Extraction of Entities and Overlapping Relations Using Position-Attentive Sequence Labeling
			  https://www.aaai.org/ojs/index.php/AAAI/article/view/4591
			  
			  b-11 Relation Extraction Baseline System—InfoExtractor 2.0
			  https://github.com/PaddlePaddle/Research/tree/master/KG/DuIE_Baseline
			  
			  各方法的详细介绍请阅读论文或看下面的链接。
			  详见https://zhuanlan.zhihu.com/p/77868938

	- 3. 基于深度学习半监督方法

		- a. 远程监督

		  定义：假设某对实体含有某种关系，那么只要含有这对实体的句子都含有这种关系。
		  优点：可获取大量数据，无需人工标注。
		  缺点：引入了大量的噪声。
		  
		  为了缓解噪声问题，可采取多示例学习、强化学习和与训练机制。
		  详见| https://zhuanlan.zhihu.com/p/77868938

		- b. Bootstrapping自扩展方法

		  比较常见的方法有DIPRE和Snowball，相比DIPRE，Snowball通常会对获得的模板样式进行置信度计算，一定程度上可以保证抽取结果的质量。
		  
		  定义：使用少量的样本去训练一个模型，然后利用模型去抽取更多的实例，再通过新数据做迭代训练。
		  
		  优点：所需数据少，构建成本低，适合大规模的关系任务并且具备发现新关系的能力。
		  缺点：对初始样本比较敏感，存在语义漂移，结果准确率低的情况。

	- 4. 无监督方法

	  利用语料中存在的大量冗余信息做聚类，在聚类结果的基础上给定关系，但由于聚类方法本身就存在难以描述关系和低频实例召回率低的问题，因此无监督学习一般难以得到很好的抽取效果。

- 属性抽取

  由于可以把实体的属性看作实体与属性值之间的一种名词性关系，因此属性抽取任务可以转化为关系抽取任务。

	- 转化为 关系抽取任务

- 三元组抽取

  三元组，即（S, P, O），S是头实体，O是尾实体，P是两个实体之间的关系。

	- 1. 基于DGCNN和概率图的轻量级信息抽取模型

	  优势：抽取结构是苏神自己设计的CNN+Attention，足够快速。
	  https://kexue.fm/archives/6671

	- 2. 基于Bert的三元组抽取

	  https://kexue.fm/archives/7161
	  https://github.com/bojone/bert_in_keras
	  https://github.com/bojone/bert4keras/blob/master/examples/task_relation_extraction.py

- 事件抽取

  事件的发生通常包括时间、地点、参与者等属性。事件是特定时间点或时间段、特定领域范围内，由一个或多个角色参与的一个或多个动作组成的事情或状态的改变。目前已存在的知识资源（如Wiki等）所描述实体及实体间的关联关系大多是静态的，事件能描述粒度更大的、动态的、结构化的知识，是现有知识资源的重要补充。
  
  事件抽取则是从自然语言文本中抽取出用户感兴趣的事件信息，并以结构化的形式展现出来。
  
  详见|事件抽取论文和方案
  https://zhuanlan.zhihu.com/p/136433610
  https://mp.weixin.qq.com/s/CRm5ky3J-eNim90oArD6Tg

	- 子任务

		- a. 事件触发词识别

		  即识别事件类型，如‘出生’、‘离职’等

		- b. 事件元素抽取与角色分类

		  事件元素的角色通常由两部分组成，事件参与者和时间属性，事件参与者是事件的必要部分，通常是命名实体的人名和组织机构名，事件属性包括通用事件属性和事件相关属性

		- d. 事件属性标注、事件共指消解等
		- c. 事件整体特性

		  如极性（正面/负面）、语态（确定/未知）、泛型（具体/普遍）、时态（过去/现在/将来/未知）

	- 方法

		- 1. 基于模板匹配

		  a. 基于人工标注语料
		  b. 基于弱监督
		  人工标注费时费力，且存在一致性问题，而弱监督方法无需对语料完全标注，只需人工对语料进行一定的预分类或者制定种子模板，由机器根据预分类语料或种子模板自动进行模式学习。
		  优势：在特定领域中性能较好，知识表示简洁，便于理解和后续应用。
		  劣势：对于语言、领域和文档形式都有不同程度的依赖，覆盖度和可移植性较差。

		- 2. 基于统计-传统机器学习

		  主要方法为将事件类别及事件元素的识别转换为分类问题。重点在于分类器和特征的选择，常用分类算法有SVM、ME等。
		  
		  优势：与领域无关，移植性好。
		  劣势：需要大规模已标注的标准语料，否则会有严重的数据稀疏。

		- 3. 基于统计-深度学习

		  类比于传统机器学习，主要方式还是将事件抽取的各步骤转换为分类问题，不过是将分类器换成了深度学习分类算法而已。另外，也有人将事件抽取转换为序列标注和MRC问题。

### 知识融合

知识融合是面向知识服务和决策问题，以多源异构数据为基础，在本体库和规则库的支持下，通过知识抽取和转换获得隐藏在数据资源中的知识因子及其关联关系，进而在语义层次上组合、推理、创造出新知识的过程，并且这个过程需要根据数据源的变化和用户反馈进行实时动态调整。
知识融合从融合层面划分可以分为数据层知识融合和概念层数据融合。前者是指多源数据对齐，后者是指对多个知识库或信息源在概念层进行模式对齐。工具有Falcon-AO、YAM++、Dedupe等。

- 本体对齐

  本体对齐或者本体匹配是概念层知识融合的主要研究任务，是指确定本体概念之间映射关系的过程。

	- 核心

		- 如何通过本体概念之间的相似性度量，发现异构本体间的匹配关系

	- 方法

		- 基于结构、基于实例
		- 基于语言学的匹配、基于文本的匹配、基于已知本体实体联结的匹配

- 实体链接

  实体链接是数据层知识融合的主要任务，主要方法有基于实体知识的链接方法、基于篇章主题的链接方法和融合实体知识与篇章主题的链接方法。
  
  实体链接主要解决实体名的歧义性和多样性问题，是指将文本中实体名指向其所代表的真实世界实体的任务，也通常被称为实体消歧（也有人认为实体链接是实体识别和实体消歧的联合过程）。
  
  Entity Linking，难点在于两个原因，即Mention Variations（多词一义）和Entity Ambiguity（一词多义），多词一义是指同一实体有不同的mention，实体的标准名、别名、名称缩写等都可以用来指代该实体，如宋江，可以用及时雨、孝义黑三郎、宋押司来指代；一词多义是指同一mention对应不同的实体，比如苹果，可能指水果，也可能指的是苹果公司/手机/电脑。
  
  资料：
  https://zhuanlan.zhihu.com/p/100248426
  http://nlpprogress.com/english/entity_linking.html

	- 核心

		- 构建多类型多模态上下文及知识的统一表示，并建模不同信息、不同证据之间的相互交互

	- 方案

		- 1. Candidate Entity Generation(CEG, 候选实体生成)

		  最有效的方法是Name Dictionary，即词典匹配，构建实体映射词表。说白了就是配别名，如首字母缩写、模糊匹配、昵称、常见拼写错误等。
		  
		  构建方法：
		  a. 百科网站（标题、重定向页、消歧页、加粗短语/小别名、超链接）；
		  b. 基于搜索引擎：调Google API，搜mention。若前m个有wiki entity，建立map；
		  c. 直接抽取知识图谱中已有的别名；
		  d. 规则构建；
		  e. 人工标注、用户日志。
		  
		  文本与词典之间的匹配规则可分为完全匹配和模糊匹配。模糊匹配有那么几种情况，输入文本和词典词二者之间是a包含b或者b包含a的关系，则匹配成功；二者之间存在一定程度的重叠，则匹配成功；二者符合字符串相似度算法，character dice score, skip bigram dice score, hamming distance等。

		- 2. Entity Disambiguation (ED, 实体消歧)

		  候选实体排序（概率/相似度）
		  
		  	a. 利用上下文无关特征，如mention到实体的Link Count，实体自身的一些属性（比如热度、类型等）。例如，问“姚明有多高？”时，大概率是在问篮球明星姚明，而不是其他默默无闻的“姚明”。
		  
		  	b. 分类模型：根据有标注的语料，利用上下文等构建特征向量。输入是候选实体的信息和Entity Mention的特征信息、上下文信息。
		  案例|CCKS_2019 entity_link  No.1
		  https://github.com/panchunguang/ccks_baidu_entity_link
		  
		  	c. 空间向量模型：根据上下文分别构建指称实体和候选实体的特征向量，然后计算它们的余弦相似度，选取相似度最高的候选实体作为目标实体。它可以有效利用上下文信息；但空间向量为词袋模型，不能反映词语之间的语义关系，会带来维度灾难与语义隔绝的问题。
		  
		  	d. 排序模型：利用Learn to Rank(LTR)排序模型，根据查询与文档的文本相似度（余弦相似度）、欧氏距离、编辑距离、主题相似度、实体流行度等特征进行训练和预测，选取排序最高的作为目标实体。它的优势就是可以有效地融入不同的特征。
		  案例|LTR
		  https://github.com/ChenglongChen/tensorflow-LTR
		  
		  	e. 主题模型：根据指称实体与候选实体的主题分布相似度进行目标实体的确认。该方法的主要优势是能在一定程度上反映实体的语义相关性，避免维度灾难，在上下文信息比较丰富的情况下，能够取得很好的效果。

		- 3. Unlinkable Mention Prediction (无链接指代预测)

		  由于知识图谱的不完备性，会出现实体提及在知识图谱中无相对应的实体的情况，此时，对应实体应是“空实体（NIL）”。
		  
		  a. NIL Threshold：设置一个置信度阈值，如果Top_1的候选实体的预测得分小于阈值，则判定不在知识库中。
		  
		  b. Binary Classification：训练一个二分类模型，判断Top_ranked Entity是否真的是Mention表达的实体。
		  
		  c. Rank with NIL：rank的时候，候选实体中加入NIL。

### 知识存储

知识存储是针对知识图谱的知识表示形式设计底层存储方式，完成各类知识的存储，以支持对大规模图数据的有效管理和计算。知识存储的对象包括基本属性知识、关联知识、事件知识、时序知识和资源类知识等。
说白了就是如何选择数据库。从数据库层面，可选择的有图数据库、No SQL数据库、关系数据库等。

- 图数据库

	- 适用于KG结构复杂、关系复杂、连接多

- 传统关系数据库

	- 适用于KG侧重节点知识、关系简单、连接少

- NoSQL数据库

	- 考虑KG的性能、扩展性和分布式等

### 知识计算

知识计算是基于已构建的知识图谱进行能力输出的过程，它以图谱质量提升、潜在关系挖掘与补全、知识统计与知识推理作为主要研究内容。

- 知识统计与图挖掘

  知识统计与图挖掘是指基于图论的相关算法，主要包括：图查询检索、图特征统计、关联分析、时序分析、节点分类、异常检测、预测推理等。

	- 重点研究知识查询、指标统计、图挖掘

- 知识推理

  知识推理是指从知识库中已有的实体关系数据出发，进行计算机推理，建立实体间的新关联，从而拓展和丰富知识网络。知识推理是知识图谱构建的重要手段和关键环节，通过知识推理，能够从现有知识中发现新的知识。
  知识推理的对象不限于实体间的关系，也可以是实体的属性值，本体的概念层次关系等。

	- 1. 基于符号的推理

	  三元组结构，如RDF，有概念，然后基于概念符号进行推理。

	- 2. 基于OWL本体的推理

	  最常见的OWL推理工具是Jena，Jena2支持基于规则的简单推理，它的推理机制支持将推理器（inference reasoners）导入Jena，创建模型时将推理器与模型关联以实现推理。

	- 3. 基于图的方法（PRA）

	  Path Ranking算法的思想比较简单，就是以连接两个实体的已有路径作为特征构建分类器，来预测他们之间可能存在的潜在关系。它是将知识图谱视为图（以实体为节点，以关系或属性为边），从源节点开始，在图上执行随机游走或遍历，如果能够通过一个路径到达目标节点，则推测源节点和目标节点可能存在关系。
	  PRA提取特征的方法主要有随机游走、广度优先和深度优先遍历，特征值计算方法有随机游走probability，路径出现/不出现的二值特征以及路径的出现频次等。
	  优点：直观、解释性好
	  缺点：很难处理关系稀疏的数据，很难处理低连通度的图，路径特征提取的效率低且耗时

	- 4. 基于分布式知识语义表示的方法（Trans系列模型）

	  思想：将实体和关系映射到一个低维的embedding空间中，基于知识的语义表达进行推理建模。

		- TransE、TransR、TransD等

		  <1> Trans E
		  将每个词表示成向量，然后向量之间保持一种类比的关系。比如北京-中国类比巴黎-法国，即实体北京加上关系首都就等于中国，然后巴黎加上首都的关系等于法国。所以它是无限的接近于伪实体的embedding。
		  模型比较简单，但是它只能处理实体之间一对一的关系，不能处理多对一与多对多的关系。
		  
		  <2> Trans R
		  通过将实体和关系投射到不同的空间里面，解决了上面提到的一对多或者多对一、多对多的问题。
		  
		  <3> Trans H、Trans D...
		  详见|Trans系列
		  https://mp.weixin.qq.com/s/STflo3c8nyG6iHh9dEeKOQ

	- 5. 基于深度学习的推理

	  达观 https://zhuanlan.zhihu.com/p/44156544

### 知识运维

由于构建全量的行业知识图谱成本很高，在真实的场景落地过程中，一般遵循小步快走、快速迭代的原则进行知识图谱的构建和逐步演化。
知识运维是指在知识图谱初次构建完成之后，根据用户的使用反馈、不断出现的同类型知识以及增加的新的知识来源进行全量行业知识图谱的演化和完善的过程，运维过程中需要保证知识图谱的质量可控及逐步的丰富衍化。

- a. 从数据源方面的基于增量数据的知识图谱的构建过程监控

  如何持续地对图谱进行更新是一个很重要的问题。普通的图谱增量更新包括新元素的加入（节点、边或对应的属性）、旧元素属性的更改，复杂场景下可能会涉及已有元素的删除操作。
  目前有两种主要的增量方式，数据从消息队列导入图谱、利用工作流引擎定时更新图谱。

- b. 通过知识图谱的应用层发现的知识错误和新的业务需求

  如错误的实体属性值、缺失的实体间关系、未识别的实体、重复实体等。

## 图谱问答技术方案

### 基于规则模板

- 通过人工构造规则识别实体，并将问题映射到属性，填充模板生成查询语句
- 优劣势

	- 规则/模板可解释、易理解、可人工编辑、可控性好，具有较高准确率
	- 召回率较低，因为模板有限，难以覆盖多样性的自然语言问题

- 案例分析

	- a. 基于豆瓣电影/书籍的知识图谱问答

	  https://github.com/weizhixiaoyi/DouBan-KGQA
	  自然语言问句转换为SPARQL查询语句
	  <1> 问句理解
	  实体识别（构建实体词表，从问句中提取词表中所包含的实体）
	  属性链接（构建关键词集合，把问句中所包含的关键词当作问句的目标属性）
	  <2> 答案推理
	  <2.1> 基于规则：获取实体和属性之后，根据规则模板将问句转换得到查询语句
	  优势在可处理简单和复杂问句，劣势在模板构建麻烦，仅能处理已定义规则，不能覆盖问句的所有情况。
	  <2.2> 基于表示学习：Trans系列方法。以Trans E为例，知识图谱中三元组向量化后可以表示为<S, P, O>，其中S为头实体，P为关系，O为尾实体。Trans E假设实体和关系之间存在S+P≈O，即头实体S加上关系P的向量信息近似等于尾实体。那就可以通过头实体和关系预测得到尾实体，即能够通过问句中的实体和目标属性信息预测得到问句答案。

	- b. 医药领域的知识图谱问答

	  https://github.com/liuhuanyong/QASystemOnMedicalKG
	  <1> 问句的意图识别（关键词）+Mention检出（特征词）
	  <2> 将意图和Mention对应到模板，转换为查询语句

### 基于信息抽取

- 模型方案

	- 1. 基于实体与关系识别的模型

	  通过实体识别等方法获得问句中的实体Mention，再通过实体链接将Mention对应到数据库中的实体，然后得出该实体在数据库中的邻居子图，并抽取出所有的关系作为候选集，之后通过关系模型（分类或相似度匹配）得到最终关系。
	  同时，此类模型大多会加上一个问题类型识别模块以提升表现。比如，问句分为单跳和多跳问句，多跳问句在查询图结构上有多种形式，通过训练模型，能够将问句分到相应的查询图结构模板上。结合模板信息，以及之前获得的候选关系中打分高的一个（单跳问题）或多个（多跳问题）作为结果，将其与实体对应的查询路径填入后查询数据库即可得出答案。

	- 2. 基于路径匹配的模型

	  类似前面的模型，同样需要NER+EL将Mention链接到KG中的实体。但在关系模型中，不直接选取最优关系，而是直接获取所有能匹配上问题类型模板的、在中心实体周边的查询路径，之后通过各类基于问句与查询路径的特征给后者打分，选取最优的查询路径，即获得答案。

- 优劣势

	- 较高的准确率和召回率
	- 可解释性差

- 案例分析

	- a. KBQA

	  https://github.com/WenRichard/KBQA-BERT
	  单跳问题基础方案：实体识别+关系匹配-->图谱查询

	- b. 一个浙大小哥哥的项目

	  http://www.zq-ai.com/#/kgqa （体验网址）
	  https://github.com/wangjiezju1988/kgqa?from=singlemessage （刚开头）
	  
	  bert4keras electratiny 场景分类 + NER实体识别 + SimBert相似度匹配
	  五个场景：SP->O，SP-(O)-P->O，PO->S，OP->S，SO->P
	  其中S是头实体，P是关系，O是尾实体
	  SP->O：王健林他爹是谁？
	  SP-(O)-P->O：王思聪他爹创立的公司是什么？
	  PO->S：老爹是王健林的那个人是谁？
	  OP->S：王健林是谁他爹？
	  SO->P：姚明和马云谁更高？
	  分析：通过对问句进行场景分类，针对性解决个别类型的复杂问题，如对比型和事实型多跳等。

	- c. 基于多标签策略的中文知识图谱问答

	  详见https://zhuanlan.zhihu.com/p/144553222
	  https://blog.csdn.net/zzkv587/article/details/102954876

### 基于语义解析

通过对自然语言进行语义上的分析，将问句转化为一种能够让知识库’看懂’的语义表示（即逻辑形式），进而通过相应的查询语句在知识库中进行查询，从而得出答案。
语法解析的过程可以看作是自底向上构造语法树的过程，树的根节点就是该自然语言问句最终的逻辑形式表达。

- 优劣势

	- 召回比模板高，可解释性尚可
	- 涉及语法相关知识，较为麻烦

### 基于向量建模

向量建模的思想与信息抽取比较接近，前期都是通过把问题中的主题词映射到知识库中的实体，得到候选答案。基于向量建模是把问题和候选答案统一映射到一个低维空间，得到它们的分布式表达，通过数据集对该分布式表达进行训练，使得问题向量和它对应的正确答案向量在低维空间的关联得分（通常以点乘为形式）尽量高。当模型训练完成后，即可根据候选答案的向量表达和问句表达的得分进行筛选，将得分最高的作为最终答案。

- 优劣势

	- 实现简单，几乎无需人工定义特征
	- 缺少解释性

### 文本混合问答

<1> Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text  2018
论文https://zhuanlan.zhihu.com/p/65478243
链接https://github.com/OceanskySun/GraftNet

- 将文本作为补充知识源，解决数据稀疏性问题
- 优劣势

	- 更高的准确率和召回率
	- 可解释性差，适用条件严格（配对文本）

### 基于深度学习

- 1. 优化上述传统问答方案

	- 优劣势

		- 较高的准确率和召回率
		- 神经网络可解释性差

- 2. End-to-End

  将离散的问题表示为连续向量，通过深度神经网络理解问题。通常采用端到端的方式，同时建模问题表示和属性关联，直接将问题和知识图谱作为模型输入，并预测属性理解的结果，问题和知识图谱的特征表示与模型的其他参数是同时训练的。

	- 模型

		- Attention Model、Memory Network

		  <1>Large-scale Simple Question Answering with Memory Networks  2015
		  <2>Question Answering over Knowledge Base with Neural Attention Combining Global Knowledge Information  2016
		  <3>Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases  2019
		  论文https://zhuanlan.zhihu.com/p/86246631
		  链接https://github.com/hugochan/BAMnet

	- 优劣势

		- 简单问题准确率不错
		- 复杂问题效果差，难以落地

## 图谱应用

知识应用是指将知识图谱特有的应用形态与领域数据与业务场景相结合，助力领域业务转型。典型应用包括语义搜索、智能问答以及可视化决策支持等。如何针对业务需求设计实现知识图谱应用，并基于数据特点进行优化调整，是知识图谱应用的关键研究内容。

### 精准分析、语义搜索、推荐系统（用户画像、内容画像等）、智能问答（智能客服、对话系统、问答系统）、阅读理解、可视化决策支持（决策结果可解释）、语言生成（机器辅助写作/智能自动创作）等

## 研究趋势

### 知识类型与表示

- 复杂知识、复杂关系

### 知识获取

- 数据获取，质量和效率

### 知识应用

- KG与业务场景深度融合

### 知识融合

- 多源异构、多模态、多语言

## 资料推荐

### 书籍推荐

- 《知识图谱：方法、实践与应用》、《知识图谱：概念与技术》、《知识图谱与深度学习》、《知识图谱》

### 课程推荐

- 斯坦福2020春季Knowledge Graph课程 CS520

### 图谱推荐

- 中文开放知识图谱OpenKG

  中文开放知识图谱Open KG http://www.openkg.cn/
  清华K12基础教育知识图谱 http://www.edukg.cn/
  复旦知识工厂 http://kw.fudan.edu.cn
  数眼科技 http://shuyantech.com/cndbpedia/search
  唐诗别苑 http://tsby.e.bnu.edu.cn
  红楼梦 https://bluejoe2008.github.io/igraph/example1.html
  海贼王 https://mrbulb.github.io/ONEPIECE-KG/

## lk

