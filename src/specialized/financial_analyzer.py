"""
Financial Analysis Module for Computer Assistant

This module provides comprehensive financial analysis capabilities including:
- Market data analysis and visualization
- Trading signal generation and backtesting
- Portfolio management and risk assessment
- Financial news sentiment analysis
- Automated trading strategies
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'volume': self.volume,
            'market_cap': self.market_cap,
            'pe_ratio': self.pe_ratio,
            'dividend_yield': self.dividend_yield
        }


@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float]
    stop_loss: Optional[float]
    timestamp: datetime
    reasoning: str
    risk_level: RiskLevel
    expected_return: Optional[float] = None
    time_horizon: Optional[str] = None  # "short", "medium", "long"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'stop_loss': self.stop_loss,
            'timestamp': self.timestamp.isoformat(),
            'reasoning': self.reasoning,
            'risk_level': self.risk_level.value,
            'expected_return': self.expected_return,
            'time_horizon': self.time_horizon
        }


@dataclass
class PortfolioPosition:
    """Portfolio position structure"""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float  # Portfolio weight percentage


@dataclass
class FinancialAnalysis:
    """Financial analysis results"""
    symbol: str
    analysis_type: str
    timestamp: datetime
    metrics: Dict[str, float]
    signals: List[TradingSignal]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class FinancialAnalyzer:
    """
    Advanced financial analysis and trading automation system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the financial analyzer"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.analysis_params = {
            'sma_periods': [20, 50, 200],
            'ema_periods': [12, 26],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'volume_sma_period': 20,
            'min_confidence': 0.6,
            'risk_free_rate': 0.02
        }
        self.analysis_params.update(self.config.get('analysis_params', {}))
        
        # Data storage
        self.market_data: Dict[str, List[MarketData]] = {}
        self.signals_history: List[TradingSignal] = []
        self.portfolio_positions: Dict[str, PortfolioPosition] = {}
        self.watchlist: List[str] = []
        
        # Analysis cache
        self.analysis_cache: Dict[str, FinancialAnalysis] = {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        # Performance tracking
        self.performance_stats = {
            'analyses_performed': 0,
            'signals_generated': 0,
            'successful_predictions': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        # News and sentiment analysis
        self.news_sources = self.config.get('news_sources', [])
        self.sentiment_keywords = {
            'positive': ['bullish', 'growth', 'profit', 'strong', 'beat', 'exceed', 'positive'],
            'negative': ['bearish', 'decline', 'loss', 'weak', 'miss', 'below', 'negative']
        }
        
        # Risk management
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_sector_exposure': 0.3,  # 30% in any sector
            'max_daily_loss': 0.02,  # 2% daily loss limit
            'var_confidence': 0.95  # VaR confidence level
        }
        self.risk_limits.update(self.config.get('risk_limits', {}))
        
        self.logger.info("Financial analyzer initialized")
    
    async def add_market_data(self, data: MarketData) -> None:
        """Add market data for analysis"""
        try:
            if data.symbol not in self.market_data:
                self.market_data[data.symbol] = []
            
            self.market_data[data.symbol].append(data)
            
            # Keep only recent data (configurable limit)
            max_history = self.config.get('max_data_history', 1000)
            if len(self.market_data[data.symbol]) > max_history:
                self.market_data[data.symbol] = self.market_data[data.symbol][-max_history:]
            
            # Sort by timestamp
            self.market_data[data.symbol].sort(key=lambda x: x.timestamp)
            
            self.logger.debug(f"Added market data for {data.symbol}")
            
        except Exception as e:
            self.logger.error(f"Error adding market data: {e}")
    
    async def analyze_symbol(self, symbol: str, analysis_type: str = "comprehensive") -> FinancialAnalysis:
        """Perform comprehensive financial analysis on a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{analysis_type}"
            if cache_key in self.analysis_cache:
                cached_analysis = self.analysis_cache[cache_key]
                if (datetime.now() - cached_analysis.timestamp).seconds < self.cache_ttl:
                    return cached_analysis
            
            if symbol not in self.market_data or len(self.market_data[symbol]) < 50:
                raise ValueError(f"Insufficient data for {symbol}")
            
            data = self.market_data[symbol]
            df = self._create_dataframe(data)
            
            # Perform technical analysis
            technical_metrics = await self._technical_analysis(df)
            
            # Perform fundamental analysis (if data available)
            fundamental_metrics = await self._fundamental_analysis(symbol, data)
            
            # Generate trading signals
            signals = await self._generate_signals(symbol, df, technical_metrics)
            
            # Risk assessment
            risk_assessment = await self._assess_risk(symbol, df, signals)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                symbol, technical_metrics, fundamental_metrics, signals, risk_assessment
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(technical_metrics, signals, risk_assessment)
            
            # Combine all metrics
            all_metrics = {**technical_metrics, **fundamental_metrics}
            
            analysis = FinancialAnalysis(
                symbol=symbol,
                analysis_type=analysis_type,
                timestamp=datetime.now(),
                metrics=all_metrics,
                signals=signals,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis
            self.performance_stats['analyses_performed'] += 1
            
            self.logger.info(f"Completed analysis for {symbol}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            raise
    
    async def _technical_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """Perform technical analysis"""
        try:
            metrics = {}
            
            # Simple Moving Averages
            for period in self.analysis_params['sma_periods']:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
                    metrics[f'sma_{period}'] = df[f'SMA_{period}'].iloc[-1]
            
            # Exponential Moving Averages
            for period in self.analysis_params['ema_periods']:
                if len(df) >= period:
                    df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
                    metrics[f'ema_{period}'] = df[f'EMA_{period}'].iloc[-1]
            
            # RSI (Relative Strength Index)
            if len(df) >= self.analysis_params['rsi_period']:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.analysis_params['rsi_period']).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.analysis_params['rsi_period']).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                metrics['rsi'] = df['RSI'].iloc[-1]
            
            # MACD
            if len(df) >= max(self.analysis_params['macd_fast'], self.analysis_params['macd_slow']):
                ema_fast = df['close'].ewm(span=self.analysis_params['macd_fast']).mean()
                ema_slow = df['close'].ewm(span=self.analysis_params['macd_slow']).mean()
                df['MACD'] = ema_fast - ema_slow
                df['MACD_Signal'] = df['MACD'].ewm(span=self.analysis_params['macd_signal']).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                metrics['macd'] = df['MACD'].iloc[-1]
                metrics['macd_signal'] = df['MACD_Signal'].iloc[-1]
                metrics['macd_histogram'] = df['MACD_Histogram'].iloc[-1]
            
            # Bollinger Bands
            if len(df) >= self.analysis_params['bollinger_period']:
                sma = df['close'].rolling(window=self.analysis_params['bollinger_period']).mean()
                std = df['close'].rolling(window=self.analysis_params['bollinger_period']).std()
                df['BB_Upper'] = sma + (std * self.analysis_params['bollinger_std'])
                df['BB_Lower'] = sma - (std * self.analysis_params['bollinger_std'])
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma
                metrics['bb_upper'] = df['BB_Upper'].iloc[-1]
                metrics['bb_lower'] = df['BB_Lower'].iloc[-1]
                metrics['bb_width'] = df['BB_Width'].iloc[-1]
            
            # Volume analysis
            if len(df) >= self.analysis_params['volume_sma_period']:
                df['Volume_SMA'] = df['volume'].rolling(window=self.analysis_params['volume_sma_period']).mean()
                metrics['volume_ratio'] = df['volume'].iloc[-1] / df['Volume_SMA'].iloc[-1]
            
            # Price momentum
            if len(df) >= 20:
                metrics['price_momentum_5d'] = (df['close'].iloc[-1] / df['close'].iloc[-6]) - 1
                metrics['price_momentum_20d'] = (df['close'].iloc[-1] / df['close'].iloc[-21]) - 1
            
            # Volatility
            if len(df) >= 20:
                returns = df['close'].pct_change()
                metrics['volatility_20d'] = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return {}
    
    async def _fundamental_analysis(self, symbol: str, data: List[MarketData]) -> Dict[str, float]:
        """Perform fundamental analysis"""
        try:
            metrics = {}
            
            # Get latest fundamental data
            latest_data = data[-1]
            
            if latest_data.pe_ratio:
                metrics['pe_ratio'] = latest_data.pe_ratio
            
            if latest_data.dividend_yield:
                metrics['dividend_yield'] = latest_data.dividend_yield
            
            if latest_data.market_cap:
                metrics['market_cap'] = latest_data.market_cap
            
            # Calculate price-to-book ratio (if book value available)
            # This would require additional fundamental data
            
            # Revenue growth, earnings growth, etc. would require historical fundamental data
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis: {e}")
            return {}
    
    async def _generate_signals(self, symbol: str, df: pd.DataFrame, metrics: Dict[str, float]) -> List[TradingSignal]:
        """Generate trading signals based on analysis"""
        try:
            signals = []
            current_price = df['close'].iloc[-1]
            
            # RSI-based signals
            if 'rsi' in metrics:
                rsi = metrics['rsi']
                if rsi < 30:  # Oversold
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=min(0.8, (30 - rsi) / 30 + 0.5),
                        price_target=current_price * 1.05,
                        stop_loss=current_price * 0.95,
                        timestamp=datetime.now(),
                        reasoning=f"RSI oversold at {rsi:.2f}",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="short"
                    ))
                elif rsi > 70:  # Overbought
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=min(0.8, (rsi - 70) / 30 + 0.5),
                        price_target=current_price * 0.95,
                        stop_loss=current_price * 1.05,
                        timestamp=datetime.now(),
                        reasoning=f"RSI overbought at {rsi:.2f}",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="short"
                    ))
            
            # Moving average crossover signals
            if 'sma_20' in metrics and 'sma_50' in metrics:
                sma_20 = metrics['sma_20']
                sma_50 = metrics['sma_50']
                
                if current_price > sma_20 > sma_50:  # Bullish alignment
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.7,
                        price_target=current_price * 1.08,
                        stop_loss=sma_20 * 0.98,
                        timestamp=datetime.now(),
                        reasoning="Bullish moving average alignment",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="medium"
                    ))
                elif current_price < sma_20 < sma_50:  # Bearish alignment
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.7,
                        price_target=current_price * 0.92,
                        stop_loss=sma_20 * 1.02,
                        timestamp=datetime.now(),
                        reasoning="Bearish moving average alignment",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="medium"
                    ))
            
            # MACD signals
            if 'macd' in metrics and 'macd_signal' in metrics:
                macd = metrics['macd']
                macd_signal = metrics['macd_signal']
                
                if macd > macd_signal and macd > 0:  # Bullish MACD
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.65,
                        price_target=current_price * 1.06,
                        stop_loss=current_price * 0.96,
                        timestamp=datetime.now(),
                        reasoning="Bullish MACD crossover",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="medium"
                    ))
                elif macd < macd_signal and macd < 0:  # Bearish MACD
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.65,
                        price_target=current_price * 0.94,
                        stop_loss=current_price * 1.04,
                        timestamp=datetime.now(),
                        reasoning="Bearish MACD crossover",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="medium"
                    ))
            
            # Bollinger Bands signals
            if 'bb_upper' in metrics and 'bb_lower' in metrics:
                bb_upper = metrics['bb_upper']
                bb_lower = metrics['bb_lower']
                
                if current_price <= bb_lower:  # Price at lower band
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=0.6,
                        price_target=(bb_upper + bb_lower) / 2,
                        stop_loss=bb_lower * 0.98,
                        timestamp=datetime.now(),
                        reasoning="Price at Bollinger Band lower bound",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="short"
                    ))
                elif current_price >= bb_upper:  # Price at upper band
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        confidence=0.6,
                        price_target=(bb_upper + bb_lower) / 2,
                        stop_loss=bb_upper * 1.02,
                        timestamp=datetime.now(),
                        reasoning="Price at Bollinger Band upper bound",
                        risk_level=RiskLevel.MEDIUM,
                        time_horizon="short"
                    ))
            
            # Filter signals by minimum confidence
            signals = [s for s in signals if s.confidence >= self.analysis_params['min_confidence']]
            
            # Update performance stats
            self.performance_stats['signals_generated'] += len(signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    async def _assess_risk(self, symbol: str, df: pd.DataFrame, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Assess risk for the symbol and signals"""
        try:
            risk_assessment = {}
            
            # Calculate volatility
            if len(df) >= 20:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                risk_assessment['volatility'] = volatility
                
                # VaR calculation
                var_95 = np.percentile(returns, 5)
                risk_assessment['var_95'] = var_95
                
                # Maximum drawdown
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                risk_assessment['max_drawdown'] = max_drawdown
            
            # Beta calculation (would need market data)
            # risk_assessment['beta'] = self._calculate_beta(symbol, df)
            
            # Signal risk assessment
            signal_risks = []
            for signal in signals:
                signal_risk = {
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'risk_level': signal.risk_level.value,
                    'potential_loss': abs(signal.stop_loss - df['close'].iloc[-1]) / df['close'].iloc[-1] if signal.stop_loss else None
                }
                signal_risks.append(signal_risk)
            
            risk_assessment['signal_risks'] = signal_risks
            
            # Overall risk level
            if 'volatility' in risk_assessment:
                vol = risk_assessment['volatility']
                if vol < 0.2:
                    risk_assessment['overall_risk'] = RiskLevel.LOW.value
                elif vol < 0.4:
                    risk_assessment['overall_risk'] = RiskLevel.MEDIUM.value
                elif vol < 0.6:
                    risk_assessment['overall_risk'] = RiskLevel.HIGH.value
                else:
                    risk_assessment['overall_risk'] = RiskLevel.VERY_HIGH.value
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing risk: {e}")
            return {}
    
    async def _generate_recommendations(self, symbol: str, technical_metrics: Dict[str, float], 
                                      fundamental_metrics: Dict[str, float], signals: List[TradingSignal],
                                      risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate investment recommendations"""
        try:
            recommendations = []
            
            # Based on signals
            buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
            sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
            
            if buy_signals:
                avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
                recommendations.append(f"Consider buying {symbol} - {len(buy_signals)} bullish signals with average confidence {avg_confidence:.2f}")
            
            if sell_signals:
                avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
                recommendations.append(f"Consider selling {symbol} - {len(sell_signals)} bearish signals with average confidence {avg_confidence:.2f}")
            
            # Risk-based recommendations
            if 'overall_risk' in risk_assessment:
                risk_level = risk_assessment['overall_risk']
                if risk_level == RiskLevel.VERY_HIGH.value:
                    recommendations.append(f"High volatility detected for {symbol} - consider reducing position size")
                elif risk_level == RiskLevel.LOW.value:
                    recommendations.append(f"Low volatility for {symbol} - suitable for conservative portfolios")
            
            # Technical recommendations
            if 'rsi' in technical_metrics:
                rsi = technical_metrics['rsi']
                if rsi > 80:
                    recommendations.append(f"RSI extremely overbought ({rsi:.1f}) - consider taking profits")
                elif rsi < 20:
                    recommendations.append(f"RSI extremely oversold ({rsi:.1f}) - potential buying opportunity")
            
            # Fundamental recommendations
            if 'pe_ratio' in fundamental_metrics:
                pe = fundamental_metrics['pe_ratio']
                if pe > 30:
                    recommendations.append(f"High P/E ratio ({pe:.1f}) - stock may be overvalued")
                elif pe < 10:
                    recommendations.append(f"Low P/E ratio ({pe:.1f}) - potential value opportunity")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _calculate_confidence(self, technical_metrics: Dict[str, float], 
                            signals: List[TradingSignal], risk_assessment: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            confidence_factors = []
            
            # Signal confidence
            if signals:
                avg_signal_confidence = sum(s.confidence for s in signals) / len(signals)
                confidence_factors.append(avg_signal_confidence * 0.4)
            
            # Technical indicator alignment
            alignment_score = 0
            total_indicators = 0
            
            # RSI alignment
            if 'rsi' in technical_metrics:
                rsi = technical_metrics['rsi']
                if 30 <= rsi <= 70:  # Neutral zone
                    alignment_score += 0.5
                else:
                    alignment_score += 1.0  # Strong signal
                total_indicators += 1
            
            # Moving average alignment
            if 'sma_20' in technical_metrics and 'sma_50' in technical_metrics:
                sma_20 = technical_metrics['sma_20']
                sma_50 = technical_metrics['sma_50']
                if abs(sma_20 - sma_50) / sma_50 > 0.02:  # Significant difference
                    alignment_score += 1.0
                else:
                    alignment_score += 0.3
                total_indicators += 1
            
            if total_indicators > 0:
                confidence_factors.append((alignment_score / total_indicators) * 0.3)
            
            # Risk factor
            if 'overall_risk' in risk_assessment:
                risk_level = risk_assessment['overall_risk']
                if risk_level == RiskLevel.LOW.value:
                    confidence_factors.append(0.8 * 0.2)
                elif risk_level == RiskLevel.MEDIUM.value:
                    confidence_factors.append(0.6 * 0.2)
                else:
                    confidence_factors.append(0.3 * 0.2)
            
            # Data quality factor
            data_quality = min(1.0, len(self.market_data.get(technical_metrics.get('symbol', ''), [])) / 100)
            confidence_factors.append(data_quality * 0.1)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _create_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Create pandas DataFrame from market data"""
        try:
            df_data = []
            for item in data:
                df_data.append({
                    'timestamp': item.timestamp,
                    'open': item.open_price,
                    'high': item.high_price,
                    'low': item.low_price,
                    'close': item.close_price,
                    'volume': item.volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    async def backtest_strategy(self, symbol: str, start_date: datetime, end_date: datetime,
                              initial_capital: float = 10000) -> Dict[str, Any]:
        """Backtest trading strategy"""
        try:
            if symbol not in self.market_data:
                raise ValueError(f"No data available for {symbol}")
            
            # Filter data by date range
            data = [d for d in self.market_data[symbol] 
                   if start_date <= d.timestamp <= end_date]
            
            if len(data) < 50:
                raise ValueError("Insufficient data for backtesting")
            
            df = self._create_dataframe(data)
            
            # Initialize backtest variables
            capital = initial_capital
            position = 0
            trades = []
            portfolio_values = []
            
            # Run backtest
            for i in range(50, len(df)):  # Start after enough data for indicators
                current_data = data[:i+1]
                current_df = df.iloc[:i+1]
                
                # Generate signals for current point
                technical_metrics = await self._technical_analysis(current_df)
                signals = await self._generate_signals(symbol, current_df, technical_metrics)
                
                current_price = df.iloc[i]['close']
                
                # Execute trades based on signals
                for signal in signals:
                    if signal.signal_type == SignalType.BUY and position <= 0:
                        # Buy signal
                        shares_to_buy = int(capital * 0.95 / current_price)  # Use 95% of capital
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price
                            capital -= cost
                            position += shares_to_buy
                            trades.append({
                                'timestamp': df.index[i],
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': current_price,
                                'cost': cost
                            })
                    
                    elif signal.signal_type == SignalType.SELL and position > 0:
                        # Sell signal
                        revenue = position * current_price
                        capital += revenue
                        trades.append({
                            'timestamp': df.index[i],
                            'action': 'SELL',
                            'shares': position,
                            'price': current_price,
                            'revenue': revenue
                        })
                        position = 0
                
                # Calculate portfolio value
                portfolio_value = capital + (position * current_price)
                portfolio_values.append({
                    'timestamp': df.index[i],
                    'value': portfolio_value,
                    'capital': capital,
                    'position_value': position * current_price
                })
            
            # Calculate performance metrics
            final_value = portfolio_values[-1]['value']
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate Sharpe ratio
            returns = pd.Series([pv['value'] for pv in portfolio_values]).pct_change().dropna()
            sharpe_ratio = (returns.mean() * 252 - self.analysis_params['risk_free_rate']) / (returns.std() * np.sqrt(252))
            
            # Calculate maximum drawdown
            values = pd.Series([pv['value'] for pv in portfolio_values])
            running_max = values.expanding().max()
            drawdown = (values - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Win rate
            profitable_trades = 0
            total_trades = len([t for t in trades if t['action'] == 'SELL'])
            
            for i, trade in enumerate(trades):
                if trade['action'] == 'SELL' and i > 0:
                    buy_trade = trades[i-1]
                    if trade['price'] > buy_trade['price']:
                        profitable_trades += 1
            
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            backtest_results = {
                'symbol': symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_percent': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_percent': max_drawdown * 100,
                'win_rate': win_rate,
                'win_rate_percent': win_rate * 100,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'trades': trades,
                'portfolio_values': portfolio_values
            }
            
            self.logger.info(f"Backtest completed for {symbol}: {total_return:.2%} return")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            raise
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            if not self.portfolio_positions:
                return {'message': 'No positions in portfolio'}
            
            total_value = sum(pos.market_value for pos in self.portfolio_positions.values())
            total_cost = sum(pos.quantity * pos.average_cost for pos in self.portfolio_positions.values())
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # Sector allocation (would need sector data)
            positions_summary = []
            for symbol, position in self.portfolio_positions.items():
                positions_summary.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'average_cost': position.average_cost,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_percent': position.unrealized_pnl_percent,
                    'weight': position.weight
                })
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'positions_count': len(self.portfolio_positions),
                'positions': positions_summary,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {'error': str(e)}
    
    async def add_to_watchlist(self, symbol: str) -> None:
        """Add symbol to watchlist"""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            self.logger.info(f"Added {symbol} to watchlist")
    
    async def remove_from_watchlist(self, symbol: str) -> None:
        """Remove symbol from watchlist"""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            self.logger.info(f"Removed {symbol} from watchlist")
    
    async def scan_watchlist(self) -> List[Dict[str, Any]]:
        """Scan watchlist for trading opportunities"""
        try:
            opportunities = []
            
            for symbol in self.watchlist:
                try:
                    analysis = await self.analyze_symbol(symbol)
                    
                    # Filter for high-confidence signals
                    high_confidence_signals = [
                        s for s in analysis.signals 
                        if s.confidence >= 0.7
                    ]
                    
                    if high_confidence_signals:
                        opportunities.append({
                            'symbol': symbol,
                            'signals': [s.to_dict() for s in high_confidence_signals],
                            'confidence_score': analysis.confidence_score,
                            'risk_level': analysis.risk_assessment.get('overall_risk', 'unknown'),
                            'recommendations': analysis.recommendations[:3]  # Top 3 recommendations
                        })
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {symbol} in watchlist: {e}")
                    continue
            
            # Sort by confidence score
            opportunities.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning watchlist: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            'cache_size': len(self.analysis_cache),
            'watchlist_size': len(self.watchlist),
            'symbols_tracked': len(self.market_data),
            'total_data_points': sum(len(data) for data in self.market_data.values())
        }
    
    async def export_analysis(self, symbol: str, filepath: str) -> None:
        """Export analysis results to file"""
        try:
            analysis = await self.analyze_symbol(symbol)
            
            export_data = {
                'analysis': {
                    'symbol': analysis.symbol,
                    'analysis_type': analysis.analysis_type,
                    'timestamp': analysis.timestamp.isoformat(),
                    'confidence_score': analysis.confidence_score,
                    'metrics': analysis.metrics,
                    'risk_assessment': analysis.risk_assessment,
                    'recommendations': analysis.recommendations
                },
                'signals': [s.to_dict() for s in analysis.signals],
                'performance_stats': self.get_performance_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")
            raise
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(new_config)
        
        # Update analysis parameters
        if 'analysis_params' in new_config:
            self.analysis_params.update(new_config['analysis_params'])
        
        # Update risk limits
        if 'risk_limits' in new_config:
            self.risk_limits.update(new_config['risk_limits'])
        
        self.logger.info("Configuration updated")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Clear caches
            self.analysis_cache.clear()
            
            # Save performance stats if needed
            # This could be extended to save to database
            
            self.logger.info("Financial analyzer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")