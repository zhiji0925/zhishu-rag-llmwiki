/* 知枢 · 检索方式流程文档 — mermaid 深色主题初始化 */
(function () {
  if (typeof mermaid === 'undefined') return;
  mermaid.initialize({
    startOnLoad: true,
    securityLevel: 'loose',
    theme: 'base',
    themeVariables: {
      darkMode: true,
      background: '#0b0720',
      primaryColor: '#241a4d',
      primaryTextColor: '#e9e7f7',
      primaryBorderColor: '#7c6eff',
      lineColor: '#9b95c6',
      secondaryColor: '#2b2153',
      secondaryTextColor: '#e9e7f7',
      secondaryBorderColor: '#7c6eff',
      tertiaryColor: '#17102e',
      tertiaryTextColor: '#e9e7f7',
      tertiaryBorderColor: 'rgba(126,110,240,.4)',
      clusterBkg: '#150e30',
      clusterBorder: 'rgba(126,110,240,.35)',
      edgeLabelBackground: '#150e30',
      fontFamily: 'system-ui, "Microsoft YaHei", sans-serif',
      fontSize: '14px'
    }
  });
})();
