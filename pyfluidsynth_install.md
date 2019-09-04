# pyfluidsynthのインストール方法  
Mac: [ここ](http://code.lioon.net/shell/how-to-play-midi-in-commandline.html)を参照  
``` bash
$ sudo brew install fluidsynth
$ sudo port selfupdate
$ sudo port install fluidsynth
$ sudo port install generaluser-soundfont
```
  
generaluser-soundfontは`/opt/local/share/sounds/sf2/`に配置されている  
プロジェクトがactivateされている状態で`pip install pyfluidsynth`
`/.venv/lib/python3.7/site-packages/fluidsynth.py`を編集  
以下の関数をコメントアウト
- fluid_synth_get_channel_info 
- fluid_synth_set_reverb_full
- fluid_synth_set_chorus_full
- fluid_synth_get_chorus_speed_Hz
- fluid_synth_get_chorus_depth_ms
- fluid_synth_set_midi_router

これにて，fluidsynthを使用できる
```
import fluidsynth
dir(fluidsynth)
```
  
  
  
Windowsは結局ダメだった...
Windows:  [ここ](https://qiita.com/exp/items/c67a8a50e61ba63fcd1f)
- [vcpkg](https://github.com/Microsoft/vcpkg)をダウンロードして解凍
- vcpkgを任意の場所に配置し，コマンドラインで`./bootstrap-vcpkg.bat`を実行
- vcpkgへのパスを環境変数で通す
- cmd再起動で`vcpkg --help`ができるか確認
  

- [visual studio (2017)](https://docs.microsoft.com/ja-jp/visualstudio/releasenotes/vs2017-relnotes)をインストール
- C/C++環境をインストール
- visual studio installer > visual studio community 2017 > 詳細 > 変更 > 言語パック > 英語にチェック > 変更
- `vcpkg install glib`

- cmakeをインストール
- fluidsynthの最新ソースをダウンロードし解凍，任意の場所に配置
- cmake-guiでfluidsynthをビルド
    - 入力: fluidsynthのディレクトリ
    - 出力: fluidsynthのディレクトリ内にbuildディレクトリを作成しそれを指定
    - configurationではvisual studio 2017 Win 64を指定
- ここでエラーが発生し断念

- `vcpkg install fluidsynth`
- こちらも試してみて，ビルドされたものが手に入ることはわかったが，.dllをwindowsがうまく読み込んでくれず断念