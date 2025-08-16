/* index.js */
window.HELP_IMPROVE_VIDEOJS = false;

(() => {
  const onReady = (fn) => {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  };

  // 兜底样式：即使库样式被覆盖/未生效，也只显示激活项
  function injectCarouselFallbackCSS() {
    if (document.querySelector('style[data-injected="carousel-fallback"]')) return;
    const css = `
      .carousel .carousel-item { display: none; }
      .carousel .carousel-item.is-active { display: block; }
      .carousel { overflow: hidden; }
    `;
    const style = document.createElement('style');
    style.setAttribute('data-injected', 'carousel-fallback');
    style.appendChild(document.createTextNode(css));
    document.head.appendChild(style);
  }

  // 确保只有一个 is-active；若没有则默认第一项激活
  function ensureSingleActiveItem(root) {
    const items = root.querySelectorAll('.carousel-item');
    if (!items.length) return;
    let active = root.querySelector('.carousel-item.is-active');
    if (!active) {
      items[0].classList.add('is-active');
      active = items[0];
    }
    items.forEach((it) => { if (it !== active) it.classList.remove('is-active'); });
  }

  // 初始化单个轮播并返回实例
  function initCarousel(selector, options) {
    const el = document.querySelector(selector);
    if (!el || typeof bulmaCarousel === 'undefined') return null;
    ensureSingleActiveItem(el);
    try {
      const instances = bulmaCarousel.attach(selector, options);
      return (instances && instances[0]) || null;
    } catch (e) {
      console.warn('[carousel] init failed:', selector, e);
      return null;
    }
  }

  // Bulma navbar burger
  function initBurger() {
    document.querySelectorAll('.navbar-burger').forEach(($el) => {
      $el.addEventListener('click', () => {
        const target = $el.dataset.target;
        const $target = document.getElementById(target);
        $el.classList.toggle('is-active');
        if ($target) $target.classList.toggle('is-active');
      });
    });
  }

  onReady(() => {
    injectCarouselFallbackCSS();
    initBurger();

    const common = {
      slidesToScroll: 1,
      slidesToShow: 1,
      infinite: true,        // 不用写 loop，避免版本差异
      autoplay: true,
      autoplaySpeed: 5000,
      pauseOnHover: true,
      pagination: true,
      navigation: true,
      breakpoints: [{ changePoint: 768, slidesToShow: 1, slidesToScroll: 1 }],
    };

    // Figures 轮播（自动播放）
    window._figCarousel = initCarousel('#fig-carousel', common);

    // PNG 结果轮播（不自动播放）
    const pngOpts = Object.assign({}, common, { autoplay: false });
    window._pngCarousel = initCarousel('#pdf-carousel', pngOpts);

    // 作为兜底：若有其他 .carousel，统一按 common 初始化
    if (document.querySelectorAll('.carousel').length > 0) {
      bulmaCarousel.attach('.carousel', common);
    }

    // bulma-slider（如果页面有用）
    if (typeof bulmaSlider !== 'undefined') {
      bulmaSlider.attach();
    }

    // 等图片加载完再刷新一次，避免初始宽度为 0
    window.addEventListener('load', () => {
      if (window._figCarousel && typeof window._figCarousel.refresh === 'function') {
        window._figCarousel.refresh();
      }
      if (window._pngCarousel && typeof window._pngCarousel.refresh === 'function') {
        window._pngCarousel.refresh();
      }
    });
  });
})();
