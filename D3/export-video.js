const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:3000/D3/top500_at_bats.html');
  await sleep(5000); // Wait for initial load

  for (let frame = 0; frame < 120; frame++) {
    // Set currentTime for the D3 animation
    await page.evaluate((frame) => currentTime = frame * 1000 / 60, frame);
    
    // Wait for the animation to reach this frame's state
    await sleep(50); // Adjust this based on your animation speed

    let formattedFrame = frame.toString().padStart(5, '0');
    let path = __dirname + '/png/' + formattedFrame + '.png';

    await page.setViewport({width: 1920, height: 1080, deviceScaleFactor: 2});
    await page.screenshot({path}); // Consider full-page screenshots
  }

  browser.close();
})();

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
