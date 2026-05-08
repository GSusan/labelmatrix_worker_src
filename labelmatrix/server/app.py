# -*- coding: utf-8 -*-
"""
Worker HTTP服务器
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Optional
import threading
import logging


logger = logging.getLogger(__name__)


class WorkerServer:
    """Worker HTTP服务器"""

    def __init__(self, engine, port: int = 0):
        """
        Args:
            engine: 训练/推理引擎实例
            port: 监听端口，0表示自动分配
        """
        self.engine = engine
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)

        # 配置日志
        self._setup_logging()

        # 注册路由
        self._register_routes()

        # 服务器线程
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False

    def _setup_logging(self):
        """配置Flask日志"""
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)

    def _register_routes(self):
        """注册API路由"""

        @self.app.route('/status', methods=['GET'])
        def get_status():
            """获取任务状态"""
            try:
                status_data = self.engine.get_status()
                return jsonify({
                    'success': True,
                    'data': status_data
                })
            except Exception as e:
                logger.exception("Error getting status")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/stop', methods=['POST'])
        def stop_task():
            """请求停止任务"""
            try:
                success = self.engine.stop()
                return jsonify({
                    'success': success,
                    'message': 'Stop signal sent'
                })
            except Exception as e:
                logger.exception("Error stopping task")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'success': True,
                'status': 'running',
                'task_id': self.engine.task_id
            })

        @self.app.route('/metrics', methods=['GET'])
        def get_metrics():
            """获取训练指标"""
            try:
                state = self.engine.get_status()
                return jsonify({
                    'success': True,
                    'data': {
                        'metrics': state.get('metrics', {}),
                        'progress': state.get('progress', 0),
                        'status': state.get('status', 'unknown')
                    }
                })
            except Exception as e:
                logger.exception("Error getting metrics")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/logs', methods=['GET'])
        def get_logs():
            """获取训练日志"""
            try:
                # 获取查询参数
                since = request.args.get('since')
                limit = request.args.get('limit', type=int)

                # 检查engine是否支持get_logs
                if hasattr(self.engine, 'get_logs'):
                    logs = self.engine.get_logs(since=since, limit=limit)
                else:
                    logs = []

                return jsonify({
                    'success': True,
                    'data': {
                        'logs': logs,
                        'count': len(logs)
                    }
                })
            except Exception as e:
                logger.exception("Error getting logs")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/state-file', methods=['GET'])
        def get_state_from_file():
            """从状态文件读取状态（fallback机制）"""
            try:
                state = self.engine.read_state_file()
                if not state:
                    return jsonify({
                        'success': False,
                        'error': 'State file not found or empty'
                    }), 404

                return jsonify({
                    'success': True,
                    'data': state
                })
            except Exception as e:
                logger.exception("Error reading state file")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'success': False,
                'error': 'Endpoint not found'
            }), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            }), 500

    def start(self) -> int:
        """
        启动HTTP服务器

        Returns:
            实际监听的端口号
        """
        import socket

        # 如果port为0，自动分配一个可用端口
        if self.port == 0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
            sock.close()

        def run_server():
            # 禁用Flask日志输出
            import sys
            from werkzeug.serving import WSGIRequestHandler
            original_log = WSGIRequestHandler.log
            WSGIRequestHandler.log = lambda self, *args, **kwargs: None

            try:
                self.app.run(
                    host='127.0.0.1',
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            finally:
                WSGIRequestHandler.log = original_log

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True

        logger.info(f"HTTP server started on port {self.port}")

        return self.port

    def stop(self):
        """停止HTTP服务器"""
        if self.is_running:
            self.is_running = False
            logger.info("HTTP server stopped")

    def wait(self):
        """等待服务器结束"""
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join()
