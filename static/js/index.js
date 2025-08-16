/* index.js */
window.HELP_IMPROVE_VIDEOJS = false;

document.addEventListener('DOMContentLoaded', () => {
  const common = {
    slidesToScroll: 1,
    slidesToShow: 1,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
    pauseOnHover: true,
    pagination: true,
    navigation: true,
  };

  // 只按 id 初始化，避免重复 attach 所有 .carousel
  window._figCarousel = bulmaCarousel.attach('#fig-carousel', common)[0];
  window._pngCarousel = bulmaCarousel.attach('#pdf-carousel', { ...common, autoplay: false })[0];

  if (typeof bulmaSlider !== 'undefined') bulmaSlider.attach();

  // 图片加载完后刷新一次，防止初始宽度为 0
  window.addEventListener('load', () => {
    _figCarousel && _figCarousel.refresh && _figCarousel.refresh();
    _pngCarousel && _pngCarousel.refresh && _pngCarousel.refresh();
  });
});
