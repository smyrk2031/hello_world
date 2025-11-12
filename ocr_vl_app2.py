"""
PaddleOCR-VL 実験ツール - Flaskアプリケーション（堅牢版）
すべての必要なフォルダとファイルを自動作成します
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import time
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch
import einops
from transformers import AutoModelForCausalLM, AutoProcessor
import cv2
import numpy as np

# アプリケーション初期化
app = Flask(__name__)

# 設定
BASE_DIR = Path(__file__).parent.absolute()
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'ocr_vl_uploads')
app.config['RESULTS_FOLDER'] = str(BASE_DIR / 'ocr_vl_results')
app.config['LOCAL_IMAGES_FOLDER'] = str(BASE_DIR / 'local_images')
app.config['TEMPLATES_FOLDER'] = str(BASE_DIR / 'templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

def ensure_folder(folder_path):
    """
    フォルダが存在しない場合は作成する（堅牢版）
    """
    path = Path(folder_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"警告: フォルダ作成に失敗しました {folder_path}: {e}")
        return False

def ensure_file(file_path, content):
    """
    ファイルが存在しない場合は作成する
    """
    path = Path(file_path)
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"ファイルを作成しました: {file_path}")
        return True
    except Exception as e:
        print(f"警告: ファイル作成に失敗しました {file_path}: {e}")
        return False

def initialize_templates():
    """
    テンプレートファイルを初期化
    """
    templates_dir = Path(app.config['TEMPLATES_FOLDER'])
    ensure_folder(templates_dir)
    
    # ocr_vl_base.html
    base_template = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}PaddleOCR-VL 実験ツール{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 20px;
            background-color: #f5f5f5;
        }
        .option-panel {
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .result-panel {
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .timing-badge {
            display: inline-block;
            padding: 5px 10px;
            margin: 2px;
            background-color: #e3f2fd;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .comparison-table {
            font-size: 0.9em;
        }
        .image-preview {
            max-width: 100%;
            max-height: 400px;
            object-fit: contain;
        }
    </style>
    {% block extra_css %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">PaddleOCR-VL 実験ツール</a>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        {% block content %}{% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
'''
    ensure_file(templates_dir / 'ocr_vl_base.html', base_template)
    
    # ocr_vl_index.html（長いので別ファイルから読み込むか、ここに含める）
    # 簡略版を提供（完全版は元のファイルを参照）
    index_template = '''{% extends "ocr_vl_base.html" %}

{% block title %}PaddleOCR-VL 実験ツール{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <h1>PaddleOCR-VL 実験ツール</h1>
        <p class="text-muted">画像をアップロードまたはローカルフォルダから選択して、様々なオプションでOCRを実行・比較できます</p>
    </div>
</div>

<div class="row mt-4">
    <!-- 左側: 設定パネル -->
    <div class="col-md-4">
        <!-- 画像選択 -->
        <div class="option-panel">
            <h5>画像選択</h5>
            <ul class="nav nav-tabs" id="imageTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="upload-tab" data-bs-toggle="tab" data-bs-target="#upload" type="button">
                        アップロード
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="local-tab" data-bs-toggle="tab" data-bs-target="#local" type="button">
                        ローカルフォルダ
                    </button>
                </li>
            </ul>
            <div class="tab-content mt-3" id="imageTabContent">
                <div class="tab-pane fade show active" id="upload" role="tabpanel">
                    <input type="file" id="fileInput" accept="image/*" class="form-control mb-2">
                    <button class="btn btn-primary" onclick="uploadImage()">アップロード</button>
                    <div id="uploadStatus" class="mt-2"></div>
                </div>
                <div class="tab-pane fade" id="local" role="tabpanel">
                    <button class="btn btn-secondary" onclick="loadLocalImages()">画像一覧を更新</button>
                    <div id="localImages" class="mt-2" style="max-height: 300px; overflow-y: auto;"></div>
                </div>
            </div>
            <div id="selectedImage" class="mt-3"></div>
        </div>

        <!-- タスク選択 -->
        <div class="option-panel">
            <h5>タスク</h5>
            <select class="form-select" id="taskSelect">
                <option value="ocr">OCR（文字認識）</option>
                <option value="table">Table Recognition（表認識）</option>
                <option value="formula">Formula Recognition（数式認識）</option>
                <option value="chart">Chart Recognition（グラフ認識）</option>
            </select>
        </div>

        <!-- 前処理オプション -->
        <div class="option-panel">
            <h5>前処理オプション</h5>
            <div class="mb-2">
                <input type="checkbox" id="paddingEnabled" class="form-check-input">
                <label class="form-check-label" for="paddingEnabled">パディング追加（外枠対策）</label>
                <div id="paddingOptions" style="display: none; margin-left: 20px; margin-top: 5px;">
                    <label class="form-label">パディングサイズ:</label>
                    <input type="number" id="paddingSize" value="50" min="0" max="200" class="form-control form-control-sm">
                </div>
            </div>
            <div class="mb-2">
                <input type="checkbox" id="resizeEnabled" class="form-check-input">
                <label class="form-check-label" for="resizeEnabled">リサイズ</label>
                <div id="resizeOptions" style="display: none; margin-left: 20px; margin-top: 5px;">
                    <input type="number" id="resizeWidth" placeholder="幅" class="form-control form-control-sm d-inline-block" style="width: 100px;">
                    <span> x </span>
                    <input type="number" id="resizeHeight" placeholder="高さ" class="form-control form-control-sm d-inline-block" style="width: 100px;">
                </div>
            </div>
            <div class="mb-2">
                <input type="checkbox" id="contrastEnabled" class="form-check-input">
                <label class="form-check-label" for="contrastEnabled">コントラスト調整</label>
                <input type="range" id="contrastFactor" min="0.5" max="2.0" step="0.1" value="1.0" class="form-range" style="display: none;">
                <span id="contrastValue" style="display: none;">1.0</span>
            </div>
            <div class="mb-2">
                <input type="checkbox" id="brightnessEnabled" class="form-check-input">
                <label class="form-check-label" for="brightnessEnabled">明度調整</label>
                <input type="range" id="brightnessFactor" min="0.5" max="2.0" step="0.1" value="1.0" class="form-range" style="display: none;">
                <span id="brightnessValue" style="display: none;">1.0</span>
            </div>
            <div class="mb-2">
                <input type="checkbox" id="sharpnessEnabled" class="form-check-input">
                <label class="form-check-label" for="sharpnessEnabled">シャープネス調整</label>
                <input type="range" id="sharpnessFactor" min="0.0" max="2.0" step="0.1" value="1.0" class="form-range" style="display: none;">
                <span id="sharpnessValue" style="display: none;">1.0</span>
            </div>
            <div class="mb-2">
                <input type="checkbox" id="opencvEnabled" class="form-check-input">
                <label class="form-check-label" for="opencvEnabled">OpenCV処理</label>
                <div id="opencvOptions" style="display: none; margin-left: 20px; margin-top: 5px;">
                    <div><input type="checkbox" id="opencvGrayscale" class="form-check-input form-check-input-sm"> グレースケール</div>
                    <div><input type="checkbox" id="opencvDenoise" class="form-check-input form-check-input-sm"> ノイズ除去</div>
                    <div><input type="checkbox" id="opencvThreshold" class="form-check-input form-check-input-sm"> 二値化</div>
                    <div><input type="checkbox" id="opencvBlur" class="form-check-input form-check-input-sm"> ガウシアンブラー</div>
                </div>
            </div>
        </div>

        <!-- 画像分割オプション（検出漏れ対策） -->
        <div class="option-panel">
            <h5>画像分割（検出漏れ対策）</h5>
            <div class="mb-2">
                <input type="checkbox" id="splitEnabled" class="form-check-input">
                <label class="form-check-label" for="splitEnabled">画像を分割して処理</label>
                <div id="splitOptions" style="display: none; margin-left: 20px; margin-top: 5px;">
                    <label class="form-label">行数:</label>
                    <input type="number" id="splitRows" value="2" min="1" max="5" class="form-control form-control-sm mb-2">
                    <label class="form-label">列数:</label>
                    <input type="number" id="splitCols" value="2" min="1" max="5" class="form-control form-control-sm mb-2">
                    <label class="form-label">オーバーラップ率:</label>
                    <input type="number" id="splitOverlap" value="0.1" min="0" max="0.5" step="0.05" class="form-control form-control-sm">
                    <small class="text-muted">領域間の重複率（0.1 = 10%）</small>
                </div>
            </div>
        </div>

        <!-- 推論パラメータ -->
        <div class="option-panel">
            <h5>推論パラメータ</h5>
            <div class="mb-2">
                <label class="form-label">Max New Tokens</label>
                <input type="number" id="maxNewTokens" value="512" min="10" max="2048" class="form-control form-control-sm">
            </div>
            <div class="mb-2">
                <label class="form-label">Min New Tokens</label>
                <input type="number" id="minNewTokens" value="10" min="1" max="100" class="form-control form-control-sm">
            </div>
            <div class="mb-2">
                <label class="form-label">Num Beams</label>
                <input type="number" id="numBeams" value="1" min="1" max="10" class="form-control form-control-sm">
            </div>
            <div class="mb-2">
                <input type="checkbox" id="doSample" class="form-check-input">
                <label class="form-check-label" for="doSample">Do Sample</label>
            </div>
            <div class="mb-2">
                <label class="form-label">Temperature</label>
                <input type="number" id="temperature" value="1.0" min="0.1" max="2.0" step="0.1" class="form-control form-control-sm">
            </div>
            <div class="mb-2">
                <label class="form-label">Top P</label>
                <input type="number" id="topP" value="1.0" min="0.1" max="1.0" step="0.1" class="form-control form-control-sm">
            </div>
        </div>

        <!-- 実行ボタン -->
        <div class="option-panel">
            <button class="btn btn-success w-100" onclick="runOCR()" id="runBtn">OCRを実行</button>
            <div id="runStatus" class="mt-2"></div>
        </div>
    </div>

    <!-- 右側: 結果表示 -->
    <div class="col-md-8">
        <div class="result-panel">
            <h5>実行結果</h5>
            <div id="currentResult"></div>
        </div>

        <div class="result-panel">
            <h5>実行履歴</h5>
            <button class="btn btn-sm btn-secondary mb-2" onclick="loadResults()">履歴を更新</button>
            <div id="resultsList"></div>
        </div>

        <div class="result-panel">
            <h5>結果比較</h5>
            <div id="comparisonView"></div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
let selectedImage = null;
let selectedImageSource = null;
let results = [];

// 画像選択のイベントハンドラ
document.getElementById('paddingEnabled').addEventListener('change', function() {
    document.getElementById('paddingOptions').style.display = this.checked ? 'block' : 'none';
});
document.getElementById('resizeEnabled').addEventListener('change', function() {
    document.getElementById('resizeOptions').style.display = this.checked ? 'block' : 'none';
});
document.getElementById('splitEnabled').addEventListener('change', function() {
    document.getElementById('splitOptions').style.display = this.checked ? 'block' : 'none';
});
document.getElementById('contrastEnabled').addEventListener('change', function() {
    const range = document.getElementById('contrastFactor');
    const value = document.getElementById('contrastValue');
    range.style.display = this.checked ? 'block' : 'none';
    value.style.display = this.checked ? 'inline' : 'none';
    if (this.checked) updateRangeValue('contrastFactor', 'contrastValue');
});
document.getElementById('brightnessEnabled').addEventListener('change', function() {
    const range = document.getElementById('brightnessFactor');
    const value = document.getElementById('brightnessValue');
    range.style.display = this.checked ? 'block' : 'none';
    value.style.display = this.checked ? 'inline' : 'none';
    if (this.checked) updateRangeValue('brightnessFactor', 'brightnessValue');
});
document.getElementById('sharpnessEnabled').addEventListener('change', function() {
    const range = document.getElementById('sharpnessFactor');
    const value = document.getElementById('sharpnessValue');
    range.style.display = this.checked ? 'block' : 'none';
    value.style.display = this.checked ? 'inline' : 'none';
    if (this.checked) updateRangeValue('sharpnessFactor', 'sharpnessValue');
});
document.getElementById('opencvEnabled').addEventListener('change', function() {
    document.getElementById('opencvOptions').style.display = this.checked ? 'block' : 'none';
});

['contrastFactor', 'brightnessFactor', 'sharpnessFactor'].forEach(id => {
    const range = document.getElementById(id);
    const valueId = id.replace('Factor', 'Value');
    range.addEventListener('input', () => updateRangeValue(id, valueId));
});

function updateRangeValue(rangeId, valueId) {
    const range = document.getElementById(rangeId);
    const value = document.getElementById(valueId);
    value.textContent = range.value;
}

async function uploadImage() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) {
        alert('ファイルを選択してください');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const status = document.getElementById('uploadStatus');
    status.innerHTML = '<div class="alert alert-info">アップロード中...</div>';

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            selectedImage = data.filename;
            selectedImageSource = 'upload';
            displaySelectedImage(data.filename);
            status.innerHTML = '<div class="alert alert-success">アップロード完了</div>';
        } else {
            status.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        }
    } catch (error) {
        status.innerHTML = `<div class="alert alert-danger">エラー: ${error.message}</div>`;
    }
}

async function loadLocalImages() {
    const container = document.getElementById('localImages');
    container.innerHTML = '<div class="alert alert-info">読み込み中...</div>';

    try {
        const response = await fetch('/api/local_images');
        const data = await response.json();

        if (data.images.length === 0) {
            container.innerHTML = '<div class="alert alert-warning">画像が見つかりません</div>';
            return;
        }

        let html = '<div class="list-group">';
        data.images.forEach(img => {
            html += `<a href="#" class="list-group-item list-group-item-action" onclick="selectLocalImage('${img.filename}')">
                ${img.filename} <small class="text-muted">(${(img.size / 1024).toFixed(1)} KB)</small>
            </a>`;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<div class="alert alert-danger">エラー: ${error.message}</div>`;
    }
}

function selectLocalImage(filename) {
    selectedImage = filename;
    selectedImageSource = 'local';
    displaySelectedImage(filename);
}

function displaySelectedImage(filename) {
    const container = document.getElementById('selectedImage');
    container.innerHTML = `
        <div class="alert alert-info">
            <strong>選択中:</strong> ${filename}<br>
            <img src="/api/image/${filename}" class="image-preview mt-2" alt="選択画像">
        </div>
    `;
}

async function runOCR() {
    if (!selectedImage) {
        alert('画像を選択してください');
        return;
    }

    const btn = document.getElementById('runBtn');
    const status = document.getElementById('runStatus');
    const resultDiv = document.getElementById('currentResult');

    btn.disabled = true;
    btn.textContent = '実行中...';
    status.innerHTML = '<div class="alert alert-info">OCRを実行中です。しばらくお待ちください...</div>';
    resultDiv.innerHTML = '';

    // オプションを収集
    const preprocessOptions = {
        padding_enabled: document.getElementById('paddingEnabled').checked,
        padding_size: parseInt(document.getElementById('paddingSize').value) || 50,
        resize_enabled: document.getElementById('resizeEnabled').checked,
        resize_width: parseInt(document.getElementById('resizeWidth').value) || null,
        resize_height: parseInt(document.getElementById('resizeHeight').value) || null,
        contrast_enabled: document.getElementById('contrastEnabled').checked,
        contrast_factor: parseFloat(document.getElementById('contrastFactor').value),
        brightness_enabled: document.getElementById('brightnessEnabled').checked,
        brightness_factor: parseFloat(document.getElementById('brightnessFactor').value),
        sharpness_enabled: document.getElementById('sharpnessEnabled').checked,
        sharpness_factor: parseFloat(document.getElementById('sharpnessFactor').value),
        opencv_enabled: document.getElementById('opencvEnabled').checked,
        opencv_grayscale: document.getElementById('opencvGrayscale').checked,
        opencv_denoise: document.getElementById('opencvDenoise').checked,
        opencv_threshold: document.getElementById('opencvThreshold').checked,
        opencv_blur: document.getElementById('opencvBlur').checked,
    };

    const inferenceOptions = {
        max_new_tokens: parseInt(document.getElementById('maxNewTokens').value),
        min_new_tokens: parseInt(document.getElementById('minNewTokens').value),
        num_beams: parseInt(document.getElementById('numBeams').value),
        do_sample: document.getElementById('doSample').checked,
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('topP').value),
        early_stopping: true,
    };

    const splitMode = {
        enabled: document.getElementById('splitEnabled').checked,
        rows: parseInt(document.getElementById('splitRows').value) || 2,
        cols: parseInt(document.getElementById('splitCols').value) || 2,
        overlap: parseFloat(document.getElementById('splitOverlap').value) || 0.1,
    };

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                source: selectedImageSource,
                filename: selectedImage,
                task: document.getElementById('taskSelect').value,
                preprocess_options: preprocessOptions,
                inference_options: inferenceOptions,
                split_mode: splitMode
            })
        });

        const data = await response.json();
        if (response.ok) {
            status.innerHTML = '<div class="alert alert-success">実行完了</div>';
            displayResult(data.result);
            results.push(data.result);
            loadResults();
        } else {
            status.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        }
    } catch (error) {
        status.innerHTML = `<div class="alert alert-danger">エラー: ${error.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'OCRを実行';
    }
}

function displayResult(result) {
    const container = document.getElementById('currentResult');
    const timings = result.timings;
    
    let html = `
        <div class="card">
            <div class="card-header">
                <strong>${result.filename}</strong> - ${new Date(result.timestamp).toLocaleString('ja-JP')}
            </div>
            <div class="card-body">
                <h6>前処理済み画像:</h6>
                ${result.processed_image_url ? `<img src="${result.processed_image_url}" class="image-preview mb-3">` : ''}
                
                <h6>結果:</h6>
                <pre class="bg-light p-3" style="white-space: pre-wrap; max-height: 300px; overflow-y: auto;">${escapeHtml(result.result)}</pre>
                
                ${result.split_results ? `
                    <h6 class="mt-3">分割結果:</h6>
                    <div class="accordion" id="splitResultsAccordion">
                        ${result.split_results.map((sr, idx) => `
                            <div class="accordion-item">
                                <h2 class="accordion-header">
                                    <button class="accordion-button ${idx === 0 ? '' : 'collapsed'}" type="button" data-bs-toggle="collapse" data-bs-target="#split${idx}">
                                        領域 ${sr.region}
                                    </button>
                                </h2>
                                <div id="split${idx}" class="accordion-collapse collapse ${idx === 0 ? 'show' : ''}">
                                    <div class="accordion-body">
                                        <pre class="bg-light p-2" style="white-space: pre-wrap; max-height: 200px; overflow-y: auto;">${escapeHtml(sr.text)}</pre>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <h6>時間計測:</h6>
                <div>
                    <span class="timing-badge">画像読み込み: ${timings.image_loading.toFixed(3)}s</span>
                    <span class="timing-badge">前処理: ${timings.preprocessing.toFixed(3)}s</span>
                    <span class="timing-badge">入力準備: ${timings.input_preparation.toFixed(3)}s</span>
                    <span class="timing-badge">推論: ${timings.inference.toFixed(2)}s</span>
                    <span class="timing-badge">デコード: ${timings.decoding.toFixed(3)}s</span>
                    <span class="timing-badge"><strong>合計: ${timings.total.toFixed(2)}s</strong></span>
                </div>
                
                <h6 class="mt-3">設定:</h6>
                <small>
                    <strong>タスク:</strong> ${result.task}<br>
                    <strong>Max Tokens:</strong> ${result.inference_options.max_new_tokens}<br>
                    <strong>Num Beams:</strong> ${result.inference_options.num_beams}
                </small>
            </div>
        </div>
    `;
    container.innerHTML = html;
}

async function loadResults() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();
        results = data.results;

        const container = document.getElementById('resultsList');
        if (results.length === 0) {
            container.innerHTML = '<div class="alert alert-info">実行履歴がありません</div>';
            return;
        }

        let html = '<div class="list-group">';
        results.forEach(result => {
            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${result.filename}</h6>
                            <small>${new Date(result.timestamp).toLocaleString('ja-JP')}</small><br>
                            <small class="text-muted">推論時間: ${result.timings.inference.toFixed(2)}s | 合計: ${result.timings.total.toFixed(2)}s</small>
                        </div>
                        <div>
                            <button class="btn btn-sm btn-primary" onclick="viewResult('${result.id}')">詳細</button>
                            <button class="btn btn-sm btn-secondary" onclick="addToComparison('${result.id}')">比較に追加</button>
                        </div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

let comparisonResults = [];

function addToComparison(resultId) {
    const result = results.find(r => r.id === resultId);
    if (result && !comparisonResults.find(r => r.id === resultId)) {
        comparisonResults.push(result);
        updateComparisonView();
    }
}

function updateComparisonView() {
    const container = document.getElementById('comparisonView');
    if (comparisonResults.length === 0) {
        container.innerHTML = '<div class="alert alert-info">比較する結果を選択してください</div>';
        return;
    }

    let html = `
        <table class="table table-sm comparison-table">
            <thead>
                <tr>
                    <th>ファイル名</th>
                    <th>推論時間</th>
                    <th>合計時間</th>
                    <th>Max Tokens</th>
                    <th>結果（最初の100文字）</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
    `;

    comparisonResults.forEach(result => {
        html += `
            <tr>
                <td>${result.filename}</td>
                <td>${result.timings.inference.toFixed(2)}s</td>
                <td>${result.timings.total.toFixed(2)}s</td>
                <td>${result.inference_options.max_new_tokens}</td>
                <td><small>${escapeHtml(result.result.substring(0, 100))}...</small></td>
                <td>
                    <button class="btn btn-sm btn-danger" onclick="removeFromComparison('${result.id}')">削除</button>
                </td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function removeFromComparison(resultId) {
    comparisonResults = comparisonResults.filter(r => r.id !== resultId);
    updateComparisonView();
}

function viewResult(resultId) {
    const result = results.find(r => r.id === resultId);
    if (result) {
        displayResult(result);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 初期化
loadLocalImages();
loadResults();
</script>
{% endblock %}
'''
    ensure_file(templates_dir / 'ocr_vl_index.html', index_template)

def initialize_all_folders():
    """
    すべての必要なフォルダを初期化
    """
    folders = [
        app.config['UPLOAD_FOLDER'],
        app.config['RESULTS_FOLDER'],
        app.config['LOCAL_IMAGES_FOLDER'],
        app.config['TEMPLATES_FOLDER'],
    ]
    
    print("必要なフォルダを初期化しています...")
    for folder in folders:
        if ensure_folder(folder):
            print(f"  ✓ {folder}")
        else:
            print(f"  ✗ {folder} (作成失敗)")

# 初期化実行
initialize_all_folders()
initialize_templates()

# グローバル変数
ocr_model = None
ocr_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """モデルをロード（遅延初期化）"""
    global ocr_model, ocr_processor
    if ocr_model is None:
        print("Loading PaddleOCR-VL model...")
        model_path = "PaddlePaddle/PaddleOCR-VL"
        ocr_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            ocr_model = ocr_model.to(device)
        ocr_model = ocr_model.eval()
        
        ocr_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Model loaded successfully")
    return ocr_model, ocr_processor

def apply_preprocessing(image, options):
    """前処理を適用"""
    img = image.copy()
    
    # パディング追加（外枠ギリギリの文字検出用）
    if options.get('padding_enabled', False):
        padding = options.get('padding_size', 50)
        from PIL import ImageOps
        img = ImageOps.expand(img, border=padding, fill='white')
    
    # リサイズ
    if options.get('resize_enabled', False):
        width = options.get('resize_width', img.width)
        height = options.get('resize_height', img.height)
        img = img.resize((width, height), Image.LANCZOS)
    
    # コントラスト調整
    if options.get('contrast_enabled', False):
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        factor = options.get('contrast_factor', 1.0)
        img = enhancer.enhance(factor)
    
    # 明度調整
    if options.get('brightness_enabled', False):
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img)
        factor = options.get('brightness_factor', 1.0)
        img = enhancer.enhance(factor)
    
    # シャープネス調整
    if options.get('sharpness_enabled', False):
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(img)
        factor = options.get('sharpness_factor', 1.0)
        img = enhancer.enhance(factor)
    
    # OpenCV処理（グレースケール、ノイズ除去など）
    if options.get('opencv_enabled', False):
        img_array = np.array(img)
        
        # グレースケール
        if options.get('opencv_grayscale', False):
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # ノイズ除去
        if options.get('opencv_denoise', False):
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        # 二値化
        if options.get('opencv_threshold', False):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            threshold_value = options.get('threshold_value', 127)
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            img_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # ガウシアンブラー
        if options.get('opencv_blur', False):
            kernel_size = options.get('blur_kernel', 5)
            # 奇数にする
            if kernel_size % 2 == 0:
                kernel_size += 1
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        img = Image.fromarray(img_array)
    
    return img

def run_ocr(image, task="ocr", inference_options=None, split_mode=None):
    """OCRを実行"""
    if inference_options is None:
        inference_options = {}
    
    model, processor = load_model()
    
    # プロンプト設定
    prompts = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }
    
    # 画像分割モード
    if split_mode and split_mode.get('enabled', False):
        return run_ocr_split(image, task, inference_options, split_mode, model, processor, prompts)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompts.get(task, "OCR:")},
            ]
        }
    ]
    
    # 入力準備
    input_prep_start = time.time()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    input_prep_time = time.time() - input_prep_start
    
    # 推論
    inference_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inference_options.get('max_new_tokens', 512),
            min_new_tokens=inference_options.get('min_new_tokens', 10),
            do_sample=inference_options.get('do_sample', False),
            num_beams=inference_options.get('num_beams', 1),
            temperature=inference_options.get('temperature', 1.0),
            top_p=inference_options.get('top_p', 1.0),
            use_cache=True,
            early_stopping=inference_options.get('early_stopping', True),
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else None,
            eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else None,
        )
    inference_time = time.time() - inference_start
    
    # デコード
    decode_start = time.time()
    result_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    decode_time = time.time() - decode_start
    
    return {
        'text': result_text,
        'timings': {
            'input_preparation': input_prep_time,
            'inference': inference_time,
            'decoding': decode_time,
            'total': input_prep_time + inference_time + decode_time
        }
    }

def run_ocr_split(image, task, inference_options, split_mode, model, processor, prompts):
    """画像を分割してOCRを実行（検出漏れ対策）"""
    split_rows = split_mode.get('rows', 2)
    split_cols = split_mode.get('cols', 2)
    overlap = split_mode.get('overlap', 0.1)  # 10%のオーバーラップ
    
    img_width, img_height = image.size
    tile_width = img_width // split_cols
    tile_height = img_height // split_rows
    
    results = []
    total_inference_time = 0
    total_input_prep_time = 0
    total_decode_time = 0
    
    for row in range(split_rows):
        for col in range(split_cols):
            # オーバーラップを考慮した座標計算
            x1 = max(0, int(col * tile_width - overlap * tile_width))
            y1 = max(0, int(row * tile_height - overlap * tile_height))
            x2 = min(img_width, int((col + 1) * tile_width + overlap * tile_width))
            y2 = min(img_height, int((row + 1) * tile_height + overlap * tile_height))
            
            tile = image.crop((x1, y1, x2, y2))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tile},
                        {"type": "text", "text": prompts.get(task, "OCR:")},
                    ]
                }
            ]
            
            input_prep_start = time.time()
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(device)
            input_prep_time = time.time() - input_prep_start
            total_input_prep_time += input_prep_time
            
            inference_start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=inference_options.get('max_new_tokens', 512),
                    min_new_tokens=inference_options.get('min_new_tokens', 10),
                    do_sample=inference_options.get('do_sample', False),
                    num_beams=inference_options.get('num_beams', 1),
                    temperature=inference_options.get('temperature', 1.0),
                    top_p=inference_options.get('top_p', 1.0),
                    use_cache=True,
                    early_stopping=inference_options.get('early_stopping', True),
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor.tokenizer, 'pad_token_id') else None,
                    eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor.tokenizer, 'eos_token_id') else None,
                )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            decode_start = time.time()
            result_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            decode_time = time.time() - decode_start
            total_decode_time += decode_time
            
            results.append({
                'region': f"({row},{col})",
                'text': result_text
            })
    
    # 結果を結合
    combined_text = "\n\n--- Region (0,0) ---\n" + results[0]['text']
    for r in results[1:]:
        combined_text += f"\n\n--- Region {r['region']} ---\n" + r['text']
    
    return {
        'text': combined_text,
        'split_results': results,
        'timings': {
            'input_preparation': total_input_prep_time,
            'inference': total_inference_time,
            'decoding': total_decode_time,
            'total': total_input_prep_time + total_inference_time + total_decode_time
        }
    }

@app.route('/')
def index():
    """メインページ"""
    # テンプレートフォルダが存在することを確認
    ensure_folder(app.config['TEMPLATES_FOLDER'])
    return render_template('ocr_vl_index.html')

@app.route('/api/local_images')
def get_local_images():
    """ローカル画像フォルダの画像一覧を取得"""
    # フォルダが存在することを確認
    ensure_folder(app.config['LOCAL_IMAGES_FOLDER'])
    
    images = []
    folder = app.config['LOCAL_IMAGES_FOLDER']
    
    if os.path.exists(folder):
        try:
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        mtime = os.path.getmtime(filepath)
                        images.append({
                            'filename': filename,
                            'size': size,
                            'modified': datetime.fromtimestamp(mtime).isoformat()
                        })
        except Exception as e:
            print(f"ローカル画像一覧取得エラー: {e}")
    
    return jsonify({'images': images})

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """画像をアップロード"""
    # フォルダが存在することを確認
    ensure_folder(app.config['UPLOAD_FOLDER'])
    
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/api/image/{filename}'
        })
    except Exception as e:
        return jsonify({'error': f'アップロードに失敗しました: {str(e)}'}), 500

@app.route('/api/image/<filename>')
def get_image(filename):
    """画像を返す"""
    # フォルダが存在することを確認
    ensure_folder(app.config['UPLOAD_FOLDER'])
    ensure_folder(app.config['LOCAL_IMAGES_FOLDER'])
    
    # アップロードフォルダから探す
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
    # ローカルフォルダから探す
    filepath = os.path.join(app.config['LOCAL_IMAGES_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_from_directory(app.config['LOCAL_IMAGES_FOLDER'], filename)
    
    return jsonify({'error': '画像が見つかりません'}), 404

@app.route('/api/run', methods=['POST'])
def run_ocr_api():
    """OCRを実行"""
    # フォルダが存在することを確認
    ensure_folder(app.config['UPLOAD_FOLDER'])
    ensure_folder(app.config['LOCAL_IMAGES_FOLDER'])
    ensure_folder(app.config['RESULTS_FOLDER'])
    
    data = request.json
    
    # 画像の読み込み
    if data.get('source') == 'upload':
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    elif data.get('source') == 'local':
        image_path = os.path.join(app.config['LOCAL_IMAGES_FOLDER'], data['filename'])
    else:
        return jsonify({'error': '無効な画像ソース'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': '画像ファイルが見つかりません'}), 404
    
    try:
        # 画像読み込み
        image_load_start = time.time()
        image = Image.open(image_path).convert("RGB")
        image_load_time = time.time() - image_load_start
        
        # 前処理
        preprocess_start = time.time()
        preprocess_options = data.get('preprocess_options', {})
        processed_image = apply_preprocessing(image, preprocess_options)
        preprocess_time = time.time() - preprocess_start
        
        # OCR実行
        task = data.get('task', 'ocr')
        inference_options = data.get('inference_options', {})
        split_mode = data.get('split_mode', {})
        
        ocr_result = run_ocr(processed_image, task, inference_options, split_mode)
        
        # 結果を保存
        result_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data['filename']}"
        result_data = {
            'id': result_id,
            'filename': data['filename'],
            'source': data['source'],
            'task': task,
            'preprocess_options': preprocess_options,
            'inference_options': inference_options,
            'split_mode': split_mode,
            'result': ocr_result['text'],
            'split_results': ocr_result.get('split_results'),  # 分割結果があれば保存
            'timings': {
                'image_loading': image_load_time,
                'preprocessing': preprocess_time,
                **ocr_result['timings']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果をJSONファイルに保存
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 前処理済み画像も保存
        processed_image_path = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}_processed.png")
        processed_image.save(processed_image_path)
        result_data['processed_image_url'] = f'/api/result_image/{result_id}_processed.png'
        
        return jsonify({
            'success': True,
            'result': result_data
        })
        
    except Exception as e:
        import traceback
        error_msg = f'エラーが発生しました: {str(e)}'
        print(f"OCR実行エラー: {error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/api/results')
def get_results():
    """実行結果一覧を取得"""
    # フォルダが存在することを確認
    ensure_folder(app.config['RESULTS_FOLDER'])
    
    results = []
    folder = app.config['RESULTS_FOLDER']
    
    if os.path.exists(folder):
        try:
            for filename in os.listdir(folder):
                if filename.endswith('.json'):
                    filepath = os.path.join(folder, filename)
                    if os.path.isfile(filepath):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            results.append(result)
        except Exception as e:
            print(f"結果一覧取得エラー: {e}")
    
    # タイムスタンプでソート（新しい順）
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return jsonify({'results': results})

@app.route('/api/result_image/<filename>')
def get_result_image(filename):
    """結果画像を返す"""
    # フォルダが存在することを確認
    ensure_folder(app.config['RESULTS_FOLDER'])
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/api/result/<result_id>')
def get_result(result_id):
    """特定の結果を取得"""
    # フォルダが存在することを確認
    ensure_folder(app.config['RESULTS_FOLDER'])
    
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({'error': '結果が見つかりません'}), 404

if __name__ == '__main__':
    print("=" * 60)
    print("PaddleOCR-VL 実験ツール（堅牢版）")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"作業ディレクトリ: {BASE_DIR}")
    print(f"アップロードフォルダ: {app.config['UPLOAD_FOLDER']}")
    print(f"結果フォルダ: {app.config['RESULTS_FOLDER']}")
    print(f"ローカル画像フォルダ: {app.config['LOCAL_IMAGES_FOLDER']}")
    print(f"テンプレートフォルダ: {app.config['TEMPLATES_FOLDER']}")
    print("=" * 60)
    print("Starting PaddleOCR-VL Experiment Tool...")
    print("ブラウザで http://localhost:5001 にアクセスしてください")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)

