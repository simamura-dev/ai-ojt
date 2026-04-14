/**
 * Video OCR + Claude モバイルアプリ
 * ===================================
 * スマホカメラでPC画面を撮影 → ローカルPCのFastAPIサーバーに送信
 * → Claude解析結果 + 次にすべきこと + 追加質問 の対話型UI
 *
 * 起動前:
 *   1. BACKEND_URL をローカルPCのIPに書き換える
 *   2. npm install
 *   3. npx expo start
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ScrollView,
  ActivityIndicator,
  Alert,
  SafeAreaView,
  StatusBar,
  Switch,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

// ⚠️ あなたのPCのIPアドレスに変更してください
const BACKEND_URL = 'http://192.168.1.10:8000';
const AUTO_CAPTURE_INTERVAL_MS = 5000;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);

  const [prompt, setPrompt] = useState('画面に表示されているコードやエラーを分析して');
  const [isAutoMode, setIsAutoMode] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState('待機中');

  const autoTimerRef = useRef(null);
  const processingRef = useRef(false);

  useEffect(() => {
    if (isAutoMode) {
      setStatus(`自動撮影中 (${AUTO_CAPTURE_INTERVAL_MS / 1000}秒間隔)`);
      autoTimerRef.current = setInterval(() => {
        if (!processingRef.current) captureAndAnalyze();
      }, AUTO_CAPTURE_INTERVAL_MS);
    } else {
      if (autoTimerRef.current) {
        clearInterval(autoTimerRef.current);
        autoTimerRef.current = null;
      }
      setStatus('待機中');
    }
    return () => {
      if (autoTimerRef.current) clearInterval(autoTimerRef.current);
    };
  }, [isAutoMode]);

  if (!permission) return <View style={styles.center}><ActivityIndicator /></View>;
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>カメラへのアクセス許可が必要です</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>許可する</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ─── 撮影＆解析 ───
  const captureAndAnalyze = async () => {
    if (processingRef.current || !cameraRef.current) return;
    processingRef.current = true;
    setIsProcessing(true);
    setStatus('撮影中...');

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        base64: true,
      });

      setStatus('解析 + 次のアクション生成中...');

      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: photo.base64,
          prompt: prompt || null,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setResults(prev => [{
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        ocr: data.ocr_text,
        analysis: data.analysis,
        nextSteps: data.next_steps,
        elapsed: data.elapsed_ms,
        followups: [],            // 追加質問のやり取り
        followupInput: '',
        followupLoading: false,
      }, ...prev].slice(0, 20));

      setStatus(`完了 (${data.elapsed_ms}ms)`);
    } catch (e) {
      console.error(e);
      setStatus(`エラー: ${e.message}`);
      Alert.alert('エラー', `接続失敗: ${BACKEND_URL}\n\n${e.message}`);
    } finally {
      processingRef.current = false;
      setIsProcessing(false);
    }
  };

  // ─── 追加質問 ───
  const askFollowup = async (resultId) => {
    const target = results.find(r => r.id === resultId);
    if (!target || !target.followupInput.trim()) return;

    const question = target.followupInput;

    // UIに即時反映
    setResults(prev => prev.map(r =>
      r.id === resultId
        ? { ...r, followupLoading: true, followupInput: '' }
        : r
    ));

    try {
      const response = await fetch(`${BACKEND_URL}/followup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ocr_text: target.ocr,
          previous_analysis: target.analysis || target.nextSteps,
          question,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setResults(prev => prev.map(r =>
        r.id === resultId
          ? {
            ...r,
            followupLoading: false,
            followups: [...r.followups, { question, answer: data.answer }],
          }
          : r
      ));
    } catch (e) {
      Alert.alert('エラー', e.message);
      setResults(prev => prev.map(r =>
        r.id === resultId ? { ...r, followupLoading: false } : r
      ));
    }
  };

  const updateFollowupInput = (resultId, text) => {
    setResults(prev => prev.map(r =>
      r.id === resultId ? { ...r, followupInput: text } : r
    ));
  };

  const clearResults = () => {
    setResults([]);
    setStatus('履歴をクリアしました');
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        {/* カメラ */}
        <View style={styles.cameraContainer}>
          <CameraView ref={cameraRef} style={styles.camera} facing="back" />
          <View style={styles.statusBar}>
            <Text style={styles.statusText}>{status}</Text>
            {isProcessing && <ActivityIndicator color="#fff" size="small" />}
          </View>
        </View>

        {/* コントロール */}
        <View style={styles.controls}>
          <TextInput
            style={styles.promptInput}
            value={prompt}
            onChangeText={setPrompt}
            placeholder="Claude への質問・指示"
            placeholderTextColor="#888"
            multiline
          />
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.button, isProcessing && styles.disabled]}
              onPress={captureAndAnalyze}
              disabled={isProcessing}
            >
              <Text style={styles.buttonText}>📸 撮影＆解析</Text>
            </TouchableOpacity>
            <View style={styles.autoToggle}>
              <Text style={styles.label}>自動</Text>
              <Switch value={isAutoMode} onValueChange={setIsAutoMode} />
            </View>
          </View>
          <TouchableOpacity style={styles.clearButton} onPress={clearResults}>
            <Text style={styles.clearText}>履歴クリア</Text>
          </TouchableOpacity>
        </View>

        {/* 結果リスト */}
        <ScrollView style={styles.results} keyboardShouldPersistTaps="handled">
          {results.length === 0 && (
            <Text style={styles.emptyText}>
              PC画面を撮影すると、解析結果と次のアクションが表示されます
            </Text>
          )}

          {results.map(r => (
            <View key={r.id} style={styles.resultCard}>
              <Text style={styles.resultTime}>{r.timestamp} ({r.elapsed}ms)</Text>

              {/* 分析結果 */}
              {r.analysis && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>🧠 Claudeの分析</Text>
                  <Text style={styles.bodyText}>{r.analysis}</Text>
                </View>
              )}

              {/* ✨ 次にすべきこと（メイン） */}
              {r.nextSteps && (
                <View style={styles.nextStepsSection}>
                  <Text style={styles.nextStepsTitle}>✨ 次にやるべきこと</Text>
                  <Text style={styles.bodyText}>{r.nextSteps}</Text>
                </View>
              )}

              {/* 追加質問のやり取り */}
              {r.followups.map((f, idx) => (
                <View key={idx} style={styles.followupBlock}>
                  <Text style={styles.questionText}>❓ {f.question}</Text>
                  <Text style={styles.answerText}>{f.answer}</Text>
                </View>
              ))}

              {/* 追加質問入力 */}
              <View style={styles.followupInputRow}>
                <TextInput
                  style={styles.followupInput}
                  value={r.followupInput}
                  onChangeText={(t) => updateFollowupInput(r.id, t)}
                  placeholder="追加で質問する..."
                  placeholderTextColor="#666"
                  multiline
                />
                <TouchableOpacity
                  style={[styles.askButton, r.followupLoading && styles.disabled]}
                  onPress={() => askFollowup(r.id)}
                  disabled={r.followupLoading || !r.followupInput?.trim()}
                >
                  {r.followupLoading
                    ? <ActivityIndicator size="small" color="#fff" />
                    : <Text style={styles.buttonText}>質問</Text>}
                </TouchableOpacity>
              </View>

              {/* OCR結果（折りたたみ風に小さく） */}
              <Text style={styles.ocrLabel}>📝 画面テキスト</Text>
              <Text style={styles.ocrText} numberOfLines={6}>{r.ocr}</Text>
            </View>
          ))}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20, backgroundColor: '#000' },
  cameraContainer: { height: 220 },
  camera: { flex: 1 },
  statusBar: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    padding: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
  },
  statusText: { color: '#fff', fontSize: 12 },
  controls: { backgroundColor: '#1a1a1a', padding: 10 },
  promptInput: {
    backgroundColor: '#2a2a2a', color: '#fff',
    padding: 10, borderRadius: 8,
    minHeight: 50, fontSize: 14, marginBottom: 8,
  },
  buttonRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  button: { backgroundColor: '#4a90e2', padding: 14, borderRadius: 8, alignItems: 'center', flex: 1 },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 14 },
  disabled: { opacity: 0.5 },
  autoToggle: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  label: { color: '#fff', fontSize: 12 },
  clearButton: { alignSelf: 'flex-end', padding: 6 },
  clearText: { color: '#888', fontSize: 12 },
  message: { color: '#fff', fontSize: 16, marginBottom: 20, textAlign: 'center' },

  results: { flex: 1, backgroundColor: '#0a0a0a', padding: 10 },
  emptyText: { color: '#666', textAlign: 'center', marginTop: 40, fontSize: 14 },
  resultCard: { backgroundColor: '#1a1a1a', borderRadius: 10, padding: 12, marginBottom: 12 },
  resultTime: { color: '#888', fontSize: 11, marginBottom: 8 },

  section: { marginBottom: 12, paddingBottom: 10, borderBottomWidth: 1, borderBottomColor: '#2a2a2a' },
  sectionTitle: { color: '#4a90e2', fontSize: 13, marginBottom: 6, fontWeight: '700' },

  // 次にすべきこと（強調表示）
  nextStepsSection: {
    marginBottom: 12, padding: 10,
    backgroundColor: '#1c2a1c',
    borderRadius: 8,
    borderLeftWidth: 3, borderLeftColor: '#4ade80',
  },
  nextStepsTitle: { color: '#4ade80', fontSize: 14, fontWeight: '700', marginBottom: 6 },

  bodyText: { color: '#fff', fontSize: 14, lineHeight: 20 },

  followupBlock: {
    marginBottom: 10, paddingLeft: 10,
    borderLeftWidth: 2, borderLeftColor: '#555',
  },
  questionText: { color: '#ffd166', fontSize: 13, marginBottom: 4, fontWeight: '600' },
  answerText: { color: '#e5e5e5', fontSize: 14, lineHeight: 20 },

  followupInputRow: { flexDirection: 'row', gap: 8, marginTop: 8, alignItems: 'flex-end' },
  followupInput: {
    flex: 1, backgroundColor: '#2a2a2a', color: '#fff',
    padding: 10, borderRadius: 6, fontSize: 13, minHeight: 40,
  },
  askButton: {
    backgroundColor: '#4a90e2', paddingHorizontal: 16, paddingVertical: 10,
    borderRadius: 6, justifyContent: 'center', minWidth: 60, alignItems: 'center',
  },

  ocrLabel: { color: '#555', fontSize: 10, marginTop: 10, marginBottom: 2 },
  ocrText: { color: '#666', fontSize: 11, fontFamily: 'Courier', lineHeight: 16 },
});
