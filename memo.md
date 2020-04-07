Detectron2習得

get_cfg()でconfigファイルからCfgNodeクラスを読見込む
ほとんどの関数はそれを引数に取り，パラメータを設定する

plain_train_net.py
こちらをまず見るほうがわかりやすい

- build_model(cfg)
  - モデルを生成
  - cfg.META_ARCHITECUREで概略のモデルを読む
  - GeneralizedRCNN
	- forward()でlossを返す仕様
	- inference()

- 分散処理のためのDistributedDataParallelクラス
  - モデルを分散表現に変換するものらしい
  - とりあえず関係なし


- build_detection_evaluator	
  - model.train()
    - モデルを訓練モードにする

- build_optimizer(cfg, model)
  - cfgに応じてoptimizerを作成
  - 中身を見ると，optim.SGDを使ってい
  - cfg.SOLVER.BASE_LRがベースのLearning Rate, 
  - cfg.SOLVER.MOMENTUMがモーメント
  
- build_scheduler(cfg, optimizer)
  - Learning Rateなどのスケジューリングを設定する
  - 現状では2つのスケジューラが選べるのみ
	- WarmupMultiStepLR
	- WarmupCosineLR
	
- これらはpytorch的にはoptim.SGD(param)とかで自分で設定する
- cfgでかんたんに変えられるように，というラッパー

- DetectionCheckpointer(model, output_dir, optimizer, scheduler)
  - Checkpointerからの派生クラス
  - 学習の記録とか再開とか

- PeriodicCheckpointer()
  - fvcoreのクラス
  - 定期的にcheckpointをsaveする

- writers(ログ出力)
  - CommonMetricPrinter
    - 共通の指標(iteration time, ETA, memory ,all losses and learning rate)を表示
  - JSONWriter
	- jsonファイルにevanestorageの内容を書き出す
  - TensorboardXWrte
	- tensorboardサポート
- EventStorage
  - step()で次のステップに移行
	
- build_detection_train_loader(cfg)
  - 訓練用のDataLoaderインスタンスを生成

- build_detection_test_loader
  - DataLoaderのインスタンスを作成する
  - DataLoaderについてはpytorch本にも解説あり

- loss_dict = model(data)
  - ネットワークの適用
  - training == Trueのとき，lossを返す仕様になっている
  - training != Trueのときはinferenceの結果を返す
	- GeneralizedRCNN, SemanticSegmentorなど cfg.MODEL.META_ARCHITECUREのモデルの仕様っぽい
	
- DefaultPredictor
  - predictするためのクラス

- inference_on_dataset


- train_net.py
  - 学習用のスクリプト
  - いろいろとラッピングされている

- Trainerを定義
  - DefaultTrainer 
	- とりあえずいじりたくない人のためのTrainerクラス
	- configファイルをもとに，model, optimizer, scheduler, dataloaderを作成
	- cfg.MODEL.WEIGHTを読む（途中から再開）

	- build_evaluator
	  - datasetに対するevaluatorを作成するメソッド
	- build_test_loader
	  - テスト用のDataLoaderの生成
	- build_train_loader
	  - 訓練用のDataLoaderの生成

  - SimpleTrainer
	- より色々といじりたい人のためのシンプルなTrainerクラス


- tools/train_net.pyよりDensePose/tarin_net.pyのほうが，かなり簡略化されていてよい

# MultiMaskの実装

- ROI_HEADS: headの集まりのクラス
  - StandardROIHeads: 各headに独立にroiを与えるクラス，MaskRCNNとかはこれでいい
  - 必要なら自分で書く必要はある

- detectron2/modeling/roi_heads/mask_head.pyが同じレイヤに当たる

```
  ROI_MASK_HEAD:
    NAME: "MaskRCNNConvUpsampleHead"
    NUM_CONV: 4
    POOLER_RESOLUTION: 14
```

を，

```
  ROI_MULTI_MASK_HEAD:
    NAME: "MultiMaskRCNNConvUpsampleHead"
	NUM_MASK: 2
    NUM_CONV: 4
    POOLER_RESOLUTION: 14
```

みたいにできると理想. Dataset, DataLoaderも追加する必要はある．
というわけで，mask_head.pyを読み解く．


## class

- BaseMaskRCNNHead
  - 基本のMaskRCNNのクラス
  
  - VIS_PERIOD
    - visualizationの周期(in steps)
	
```

def forward(self, x, instances: List[Instances]):

```
	- 訓練モードではmask_rcnn_loss()を返す
	- 推論モードではmask_rcnn_inference(x, instances)を返す
	
### mask_rcnn_loss(x, instances, vis_period=0)

x: Tensor in (B, C, Hmask, Wmask)

B: 画像中のMaskの数
C: foreground classの数
Hmask, Wmask: マスクの高さと幅

それぞれの要素の値はlogits(0~1の確率)

instances: list[Instances]: N個のInstanceのList, Nはバッチの数．ground truth(class, box, mask, ...)を含んでいる.

Returns: mask_loss (Tensor): A scalar tensor containing the loss.


mask_rcnn_inference(pred_mask_logits, pred_instances)

pred_mask_logits:
lossのものと同じ

pred_instances (list[Instances])
Instanceのリスト．リストの長さはバッチサイズ．Instanceは，pred_classesフィールドを持つ．
各Instanceのpred_masksフィールドに，推定されたmaskが入る. sizeは(Hmask, Wmask)


推論のクラス(maskの生成)

- MaskRCNNConvUpsampleHead(BaseMaskRCNNHead)
  - UpsamplingつきのMaskRCNNクラス
  - Conv2d + upsampling


Instanceのgt_maskメンバがMaskRCNNのGround Truthになっている．

MultiMaskの場合，Instanceにgt_multimaskメンバを追加して，

gt_multimask: {'primary': Polygon(), 'secondary': Polygon()}

みたいにするのが良いかも．とすると，mask_head.pyをコピ
ーして,multimask_head.pyを実装することになるか．

MaskRCNNConvUpsampleHeadでは，

- num_conv個のConv2d(relu付き)をかける(mask_fcn1, mask_fcn2, ...)
- ConvTranspose2dでUpsampleする
- predictor(1x1のConv2d)でclass数分のチャンネルのmaskに変換する

layers()が出力を返して，forward()がlossを返す仕様になっている
