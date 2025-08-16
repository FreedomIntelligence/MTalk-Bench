/* index.js */
window.HELP_IMPROVE_VIDEOJS = false;

document.addEventListener('DOMContentLoaded', () => {
  if (typeof bulmaCarousel === 'undefined') {
    document.querySelectorAll('.carousel').forEach(el => el.setAttribute('data-fallback', 'true'));
    if (typeof bulmaSlider !== 'undefined') bulmaSlider.attach();
    return;
  }

  const figOpts = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,             
    navigation: true,
    pagination: true,
    autoplay: true,
    autoplaySpeed: 5000,
    pauseOnHover: true,
  };

  const pdfOpts = { ...figOpts, autoplay: false };
  window._figCarousel = bulmaCarousel.attach('#fig-carousel', figOpts)[0];
  window._pdfCarousel = bulmaCarousel.attach('#pdf-carousel', pdfOpts)[0];

  if (typeof bulmaSlider !== 'undefined') bulmaSlider.attach();

  window.addEventListener('load', () => {
    _figCarousel && _figCarousel.refresh && _figCarousel.refresh();
    _pdfCarousel && _pdfCarousel.refresh && _pdfCarousel.refresh();

    console.log('fig items =', document.querySelectorAll('#fig-carousel .carousel-item').length);
    console.log('pdf items =', document.querySelectorAll('#pdf-carousel .carousel-item').length);
  });
});
