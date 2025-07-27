import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from typing import List, Dict, Tuple, Union

# 设置页面配置
st.set_page_config(
    page_title="ETF组合回测工具",
    page_icon="📈",
    layout="wide"
)

# 设置标题
st.title("ETF组合回测工具")
st.markdown("---")

# 获取ETF列表 - 使用新的同花顺接口
@st.cache_data(ttl=3600)
def get_etf_list():
    """获取ETF基金列表 - 使用同花顺理财接口"""
    try:
        # 使用同花顺接口fund_etf_spot_ths替代原接口fund_etf_spot_em
        # 目标地址: https://fund.10jqka.com.cn/datacenter/jz/kfs/etf/
        df = ak.fund_etf_spot_ths(date="")  # date=""返回最新数据
        # 只保留必要的列 - 接口返回"基金代码"和"基金名称"列
        df = df[['基金代码', '基金名称']]
        # 重命名列以保持与原代码兼容
        df = df.rename(columns={'基金代码': '代码', '基金名称': '名称'})
        return df
    except Exception as e:
        st.error(f"获取ETF列表失败: {e}")
        return pd.DataFrame(columns=['代码', '名称'])

# 获取ETF历史数据
@st.cache_data(ttl=3600)
def get_etf_hist(symbol: str, start_date: str, end_date: str):
    """获取ETF历史数据"""
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", 
                                start_date=start_date.replace('-', ''), 
                                end_date=end_date.replace('-', ''),
                                adjust="hfq")  # 使用后复权数据
        # 重命名列
        df = df.rename(columns={'日期': 'date', '收盘': 'close'})
        # 将日期列转换为日期类型
        df['date'] = pd.to_datetime(df['date'])
        # 设置日期为索引
        df = df.set_index('date')
        # 只保留收盘价
        df = df[['close']]
        return df
    except Exception as e:
        st.error(f"获取ETF {symbol} 历史数据失败: {e}")
        return pd.DataFrame()

# 计算回测指标
def calculate_metrics(portfolio_returns: pd.Series) -> Dict:
    """计算回测指标"""
    # 累计收益率
    cumulative_return = (portfolio_returns + 1).prod() - 1
    
    # 年化收益率 (假设252个交易日)
    years = len(portfolio_returns) / 252
    annual_return = (1 + cumulative_return) ** (1 / years) - 1
    
    # 年化波动率
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # 计算最大回撤
    cum_returns = (1 + portfolio_returns).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    
    # 夏普比率 (假设无风险利率为3%)
    risk_free_rate = 0.03
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
    
    # 计算卡玛比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 计算索提诺比率
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
    
    # 计算胜率
    win_rate = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns)
    
    return {
        "累计收益率": cumulative_return,
        "年化收益率": annual_return,
        "年化波动率": annual_volatility,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe_ratio,
        "卡玛比率": calmar_ratio,
        "索提诺比率": sortino_ratio,
        "胜率": win_rate
    }

# 生成回测结论
def generate_conclusion(metrics: Dict, etf_names: List[str]) -> str:
    """生成回测结论"""
    conclusion = []
    
    # 收益分析
    if metrics["累计收益率"] > 0:
        conclusion.append(f"在回测期间内，该ETF组合取得了{metrics['累计收益率']:.2%}的累计收益，年化收益率为{metrics['年化收益率']:.2%}。")
    else:
        conclusion.append(f"在回测期间内，该ETF组合出现了{metrics['累计收益率']:.2%}的累计亏损，年化收益率为{metrics['年化收益率']:.2%}。")
    
    # 风险分析
    conclusion.append(f"组合的年化波动率为{metrics['年化波动率']:.2%}，最大回撤为{abs(metrics['最大回撤']):.2%}。")
    
    # 风险调整收益分析
    if metrics["夏普比率"] > 1:
        conclusion.append(f"夏普比率为{metrics['夏普比率']:.2f}，表明组合的风险调整收益较好。")
    elif metrics["夏普比率"] > 0:
        conclusion.append(f"夏普比率为{metrics['夏普比率']:.2f}，表明组合的风险调整收益一般。")
    else:
        conclusion.append(f"夏普比率为{metrics['夏普比率']:.2f}，表明组合的风险调整收益较差。")
    
    # 综合评价
    if metrics["年化收益率"] > 0.1 and metrics["夏普比率"] > 1:
        conclusion.append("总体而言，该ETF组合表现优异，具有较高的收益和较好的风险控制能力。")
    elif metrics["年化收益率"] > 0.05 and metrics["夏普比率"] > 0.5:
        conclusion.append("总体而言，该ETF组合表现良好，收益和风险控制能力均衡。")
    elif metrics["年化收益率"] > 0:
        conclusion.append("总体而言，该ETF组合表现一般，有一定的收益但风险控制能力有待提高。")
    else:
        conclusion.append("总体而言，该ETF组合表现不佳，建议重新调整组合配置。")
    
    return " ".join(conclusion)

# 主函数
def main():
    # 侧边栏 - 参数设置
    st.sidebar.header("参数设置")
    
    # 获取ETF列表
    etf_df = get_etf_list()
    
    # 选择ETF
    selected_etfs = st.sidebar.multiselect(
        "选择ETF基金",
        options=etf_df['代码'].tolist(),
        format_func=lambda x: f"{x} - {etf_df[etf_df['代码'] == x]['名称'].values[0]}",
        help="可以选择多个ETF进行组合回测"
    )
    
    # 设置回测时间范围
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            datetime.date(2020, 1, 1),
            min_value=datetime.date(2000, 1, 1),
            max_value=datetime.date.today()
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            datetime.date.today(),
            min_value=start_date,
            max_value=datetime.date.today()
        )
    
    # 设置组合权重
    st.sidebar.subheader("组合权重设置")
    weights = {}
    if selected_etfs:
        total_weight = 0
        for etf in selected_etfs:
            etf_name = etf_df[etf_df['代码'] == etf]['名称'].values[0]
            weight = st.sidebar.slider(
                f"{etf} - {etf_name}",
                min_value=0.0,
                max_value=1.0,
                value=1.0/len(selected_etfs),
                step=0.01,
                format="%.2f"
            )
            weights[etf] = weight
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            for etf in weights:
                weights[etf] = weights[etf] / total_weight
    
    # 开始回测按钮
    start_backtest = st.sidebar.button("开始回测")
    
    # 主界面
    if not selected_etfs:
        st.info("请在侧边栏选择至少一个ETF基金进行回测")
        return
    
    if start_backtest:
        with st.spinner("正在进行回测计算..."):
            # 获取所有选中ETF的历史数据
            all_data = {}
            etf_names = {}
            for etf in selected_etfs:
                etf_name = etf_df[etf_df['代码'] == etf]['名称'].values[0]
                etf_names[etf] = etf_name
                df = get_etf_hist(etf, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if not df.empty:
                    all_data[etf] = df
            
            if not all_data:
                st.error("获取ETF数据失败，请重试")
                return
            
            # 合并所有ETF数据
            merged_data = pd.concat([df.rename(columns={'close': etf}) for etf, df in all_data.items()], axis=1)
            merged_data = merged_data.dropna()  # 删除有缺失值的行
            
            if merged_data.empty:
                st.error("合并后的数据为空，请选择其他ETF或调整时间范围")
                return
            
            # 计算每日收益率
            returns_data = merged_data.pct_change().dropna()
            
            # 计算组合收益率
            portfolio_returns = pd.Series(0, index=returns_data.index)
            for etf in selected_etfs:
                if etf in returns_data.columns:
                    portfolio_returns += returns_data[etf] * weights[etf]
            
            # 计算累计收益
            cumulative_returns = {}
            for etf in selected_etfs:
                if etf in returns_data.columns:
                    cumulative_returns[etf] = (1 + returns_data[etf]).cumprod()
            
            portfolio_cumulative_return = (1 + portfolio_returns).cumprod()
            
            # 计算回撤
            drawdowns = {}
            for etf in selected_etfs:
                if etf in returns_data.columns:
                    cum_returns = cumulative_returns[etf]
                    drawdowns[etf] = (cum_returns / cum_returns.cummax() - 1)
            
            portfolio_drawdown = (portfolio_cumulative_return / portfolio_cumulative_return.cummax() - 1)
            
            # 计算指标
            metrics = calculate_metrics(portfolio_returns)
            
            # 创建标签页
            tab1, tab2, tab3, tab4 = st.tabs(["收益走势", "最大回撤", "绩效指标", "分析结论"])
            
            with tab1:
                st.subheader("收益走势图")
                
                # 创建收益走势图
                fig = go.Figure()
                
                # 添加每个ETF的收益曲线
                for etf in selected_etfs:
                    if etf in cumulative_returns:
                        fig.add_trace(go.Scatter(
                            x=cumulative_returns[etf].index,
                            y=cumulative_returns[etf].values,
                            mode='lines',
                            name=f"{etf} - {etf_names[etf]}"
                        ))
                
                # 添加组合收益曲线
                fig.add_trace(go.Scatter(
                    x=portfolio_cumulative_return.index,
                    y=portfolio_cumulative_return.values,
                    mode='lines',
                    name='组合',
                    line=dict(width=3, dash='solid')
                ))
                
                fig.update_layout(
                    title="累计收益走势",
                    xaxis_title="日期",
                    yaxis_title="累计收益",
                    legend_title="ETF",
                    hovermode="x unified",
                    yaxis=dict(tickformat='.2%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("最大回撤曲线图")
                
                # 创建回撤曲线图
                fig = go.Figure()
                
                # 添加每个ETF的回撤曲线
                for etf in selected_etfs:
                    if etf in drawdowns:
                        fig.add_trace(go.Scatter(
                            x=drawdowns[etf].index,
                            y=drawdowns[etf].values,
                            mode='lines',
                            name=f"{etf} - {etf_names[etf]}"
                        ))
                
                # 添加组合回撤曲线
                fig.add_trace(go.Scatter(
                    x=portfolio_drawdown.index,
                    y=portfolio_drawdown.values,
                    mode='lines',
                    name='组合',
                    line=dict(width=3, dash='solid')
                ))
                
                fig.update_layout(
                    title="回撤曲线",
                    xaxis_title="日期",
                    yaxis_title="回撤",
                    legend_title="ETF",
                    hovermode="x unified",
                    yaxis=dict(tickformat='.2%')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("绩效指标")
                
                # 创建指标表格
                metrics_df = pd.DataFrame({
                    "指标": list(metrics.keys()),
                    "值": list(metrics.values())
                })
                
                # 格式化指标值
                metrics_df["格式化值"] = metrics_df.apply(
                    lambda row: f"{row['值']:.2%}" if row["指标"] in ["累计收益率", "年化收益率", "年化波动率", "最大回撤", "胜率"] 
                    else f"{row['值']:.2f}", axis=1
                )
                
                # 显示指标表格
                st.dataframe(
                    metrics_df[["指标", "格式化值"]].rename(columns={"格式化值": "值"}),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 创建指标雷达图
                radar_metrics = {
                    "年化收益率": metrics["年化收益率"],
                    "夏普比率": metrics["夏普比率"] / 3,  # 归一化处理
                    "索提诺比率": metrics["索提诺比率"] / 3,  # 归一化处理
                    "卡玛比率": metrics["卡玛比率"] / 3,  # 归一化处理
                    "胜率": metrics["胜率"],
                    "风险控制": 1 - abs(metrics["最大回撤"])  # 转换为正向指标
                }
                
                # 创建雷达图
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=list(radar_metrics.values()),
                    theta=list(radar_metrics.keys()),
                    fill='toself',
                    name='组合绩效'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("分析结论")
                
                # 生成结论
                conclusion = generate_conclusion(metrics, [etf_names[etf] for etf in selected_etfs])
                st.write(conclusion)
                
                # 显示组合权重
                st.subheader("组合权重")
                
                # 创建饼图
                fig = go.Figure(data=[go.Pie(
                    labels=[f"{etf} - {etf_names[etf]}" for etf in weights.keys()],
                    values=list(weights.values()),
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )])
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()